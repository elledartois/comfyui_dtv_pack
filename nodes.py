import json
import os
import re
import time

import numpy as np
from PIL import Image
from PIL.ExifTags import Base
from PIL.PngImagePlugin import PngInfo
from aiohttp import web

import folder_paths
from comfy.cli_args import args
from server import PromptServer


def _json_loads_maybe(value):
    if not value:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _node_title(node):
    return str(node.get("_meta", {}).get("title", "")).lower()


def _is_negative_clip_node(node):
    title = _node_title(node)
    return "negative" in title or "neg" in title


def _is_positive_clip_node(node):
    title = _node_title(node)
    return "positive" in title or "pos" in title


def _extract_clip_prompts(prompt):
    prompt = _json_loads_maybe(prompt)
    if not isinstance(prompt, dict):
        return "", ""

    clips = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        text = node.get("inputs", {}).get("text", "")
        if isinstance(text, str):
            clips.append((str(node_id), node, text))

    positive = ""
    negative = ""
    unclassified = []

    for _node_id, node, text in clips:
        if _is_negative_clip_node(node) and not negative:
            negative = text
        elif _is_positive_clip_node(node) and not positive:
            positive = text
        else:
            unclassified.append(text)

    if not positive and unclassified:
        positive = unclassified.pop(0)
    if not negative and unclassified:
        negative = unclassified.pop(0)

    return positive, negative


def _find_first_input(prompt, class_names, input_names):
    prompt = _json_loads_maybe(prompt)
    if not isinstance(prompt, dict):
        return None
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") not in class_names:
            continue
        inputs = node.get("inputs", {})
        for input_name in input_names:
            value = inputs.get(input_name)
            if not isinstance(value, list):
                return value
    return None


def _extract_size(prompt):
    width = _find_first_input(prompt, {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentVideo", "EmptyLTXVLatentVideo"}, {"width"})
    height = _find_first_input(prompt, {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentVideo", "EmptyLTXVLatentVideo"}, {"height"})
    if width is None or height is None:
        return None
    return f"{width}x{height}"


def _build_parameters(prompt):
    positive, negative = _extract_clip_prompts(prompt)
    lines = []

    if positive:
        lines.append(positive)
    if negative:
        lines.append(f"Negative prompt: {negative}")

    details = []
    steps = _find_first_input(prompt, {"KSampler", "KSamplerAdvanced"}, {"steps"})
    cfg = _find_first_input(prompt, {"KSampler", "KSamplerAdvanced"}, {"cfg"})
    sampler = _find_first_input(prompt, {"KSampler", "KSamplerAdvanced", "KSamplerSelect"}, {"sampler_name"})
    scheduler = _find_first_input(prompt, {"KSampler", "KSamplerAdvanced"}, {"scheduler"})
    seed = _find_first_input(prompt, {"KSampler", "KSamplerAdvanced"}, {"seed", "noise_seed"})
    if seed is None:
        seed = _find_first_input(prompt, {"RandomNoise", "Seed (rgthree)"}, {"noise_seed", "seed"})
    size = _extract_size(prompt)

    if steps is not None:
        details.append(f"Steps: {steps}")
    if sampler is not None:
        sampler_text = str(sampler)
        if scheduler is not None:
            sampler_text = f"{sampler_text} {scheduler}"
        details.append(f"Sampler: {sampler_text}")
    if cfg is not None:
        details.append(f"CFG scale: {cfg}")
    if seed is not None:
        details.append(f"Seed: {seed}")
    if size is not None:
        details.append(f"Size: {size}")

    if details:
        lines.append(", ".join(details))

    return "\n".join(lines).strip()


def _parse_parameters(parameters):
    if not parameters:
        return "", ""

    text = str(parameters).replace("\r\n", "\n").replace("\r", "\n")
    marker = "\nNegative prompt:"
    steps_marker = "\nSteps:"

    steps_at = text.find(steps_marker)
    prompt_area = text[:steps_at] if steps_at >= 0 else text
    negative_at = prompt_area.find(marker)

    if negative_at >= 0:
        positive = prompt_area[:negative_at].strip()
        negative = prompt_area[negative_at + len(marker):].strip()
        return positive, negative

    return prompt_area.strip(), ""


def _decode_exif_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        prefixes = [
            b"UNICODE\x00",
            b"ASCII\x00\x00\x00",
            b"JIS\x00\x00\x00\x00\x00",
        ]
        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        for encoding in ("utf-16-be", "utf-16-le", "utf-8", "latin-1"):
            try:
                return value.decode(encoding).strip("\x00").strip()
            except Exception:
                pass
        return repr(value)
    return str(value)


def _read_image_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = {}

        for key, value in img.info.items():
            if isinstance(value, bytes):
                metadata[str(key)] = _decode_exif_text(value)
            elif isinstance(value, str):
                metadata[str(key)] = value

        try:
            exif = img.getexif()
        except Exception:
            exif = None

        if exif:
            exif_map = {
                "ImageDescription": exif.get(Base.ImageDescription),
                "UserComment": exif.get(Base.UserComment),
                "Software": exif.get(Base.Software),
            }
            for key, value in exif_map.items():
                text = _decode_exif_text(value)
                if text and key not in metadata:
                    metadata[key] = text

        # JPEG output from DTVSaveImageWithMeta stores the same key/value
        # metadata as PNG in ImageDescription, wrapped as JSON.
        description = _json_loads_maybe(metadata.get("ImageDescription"))
        if isinstance(description, dict) and description.get("dtv_metadata_version") == 1:
            saved_metadata = description.get("metadata")
            if isinstance(saved_metadata, dict):
                metadata.update(saved_metadata)

        if "parameters" not in metadata:
            for key in ("UserComment", "ImageDescription", "comment", "Comment", "Description"):
                value = metadata.get(key)
                if value and ("Negative prompt:" in value or "Steps:" in value):
                    metadata["parameters"] = value
                    break

        return metadata


def _extract_prompts_from_metadata(metadata):
    parameters = metadata.get("parameters", "")
    positive, negative = _parse_parameters(parameters)

    if not positive and not negative:
        positive, negative = _extract_clip_prompts(metadata.get("prompt"))
        if not parameters:
            parameters = _build_parameters(metadata.get("prompt"))

    raw_metadata = json.dumps(metadata, ensure_ascii=False, indent=2)
    display = (
        "Positive prompt:\n"
        f"{positive}\n\n"
        "Negative prompt:\n"
        f"{negative}\n\n"
        "Parameters:\n"
        f"{parameters}"
    )

    return {
        "positive": positive,
        "negative": negative,
        "parameters": parameters,
        "raw_metadata": raw_metadata,
        "text": display,
    }


def _safe_metadata_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _expand_date_tokens(filename_prefix):
    def replace_date(match):
        fmt = match.group(1)
        replacements = {
            "yyyy": "%Y",
            "yy": "%y",
            "MM": "%m",
            "dd": "%d",
            "HH": "%H",
            "hh": "%H",
            "mm": "%M",
            "ss": "%S",
        }
        for token in sorted(replacements, key=len, reverse=True):
            fmt = fmt.replace(token, replacements[token])
        return time.strftime(fmt)

    return re.sub(r"%date:([^%]+)%", replace_date, filename_prefix)


class DTVSaveImageWithMeta:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "save_meta": ("BOOLEAN", {"default": True, "label_on": "Save meta", "label_off": "No meta"}),
                "image_format": (["png", "jpeg"], {"default": "png"}),
                "jpeg_quality": ("INT", {"default": 85, "min": 0, "max": 100, "step": 1}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "DTV Restore Prompts"

    def save_images(
        self,
        images,
        filename_prefix="ComfyUI",
        save_meta=True,
        image_format="png",
        jpeg_quality=85,
        prompt=None,
        extra_pnginfo=None,
    ):
        parameters = _build_parameters(prompt) if prompt is not None else ""
        image_format = str(image_format).lower()
        if image_format not in {"png", "jpeg"}:
            raise ValueError(f"Unsupported image format: {image_format}")
        jpeg_quality = max(0, min(100, int(jpeg_quality)))
        filename_prefix = _expand_date_tokens(filename_prefix)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata_values = {}
            if save_meta and not args.disable_metadata:
                if prompt is not None:
                    metadata_values["prompt"] = json.dumps(prompt, ensure_ascii=False)
                    if parameters:
                        metadata_values["parameters"] = parameters
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata_values[str(key)] = _safe_metadata_value(value)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            extension = "png" if image_format == "png" else "jpg"
            file = f"{filename_with_batch_num}_{counter:05}_.{extension}"
            output_path = os.path.join(full_output_folder, file)

            if image_format == "png":
                pnginfo = None
                if metadata_values:
                    pnginfo = PngInfo()
                    for key, value in metadata_values.items():
                        pnginfo.add_text(key, value)
                img.save(output_path, pnginfo=pnginfo, compress_level=self.compress_level)
            else:
                save_args = {"quality": jpeg_quality}
                if metadata_values:
                    exif = Image.Exif()
                    exif[Base.ImageDescription] = json.dumps(
                        {"dtv_metadata_version": 1, "metadata": metadata_values},
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    if parameters:
                        exif[Base.UserComment] = b"UNICODE\x00" + parameters.encode("utf-16-be")
                    exif[Base.Software] = "ComfyUI DTV Save Image"
                    save_args["exif"] = exif
                img.save(output_path, **save_args)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results, "parameters": [parameters], "text": [parameters]}, "result": (parameters,)}


class DTVReadPromptsFromPNGMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "parameters": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "parameters", "raw_metadata")
    FUNCTION = "read_prompts"
    OUTPUT_NODE = True
    CATEGORY = "DTV Restore Prompts"

    def read_prompts(self, image, parameters=""):
        image_path = folder_paths.get_annotated_filepath(image)
        parsed = _extract_prompts_from_metadata(_read_image_metadata(image_path))

        return {
            "ui": {
                "positive": [parsed["positive"]],
                "negative": [parsed["negative"]],
                "parameters": [parsed["parameters"]],
                "text": [parsed["text"]],
            },
            "result": (parsed["positive"], parsed["negative"], parsed["parameters"], parsed["raw_metadata"]),
        }


@PromptServer.instance.routes.get("/dtv_restore_prompts/read_metadata")
async def read_metadata_route(request):
    image = request.rel_url.query.get("image")
    if not image:
        return web.json_response({"error": "Missing image"}, status=400)

    try:
        image_path = folder_paths.get_annotated_filepath(image)
        parsed = _extract_prompts_from_metadata(_read_image_metadata(image_path))
        return web.json_response(parsed)
    except FileNotFoundError as err:
        return web.json_response({"error": f"File not found: {err.filename}"}, status=404)
    except Exception as err:
        return web.json_response({"error": str(err)}, status=500)


NODE_CLASS_MAPPINGS = {
    "DTVSaveImageWithMeta": DTVSaveImageWithMeta,
    "DTVReadPromptsFromPNGMetadata": DTVReadPromptsFromPNGMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DTVSaveImageWithMeta": "DTV Save Image (Save meta)",
    "DTVReadPromptsFromPNGMetadata": "DTV Read Prompts From PNG Metadata",
}
