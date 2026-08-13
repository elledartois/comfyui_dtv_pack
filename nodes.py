import json
import os
import re
import time

import numpy as np
from PIL import Image
from PIL.ExifTags import Base, IFD, TAGS
from PIL.PngImagePlugin import PngInfo
from aiohttp import web

import folder_paths
from comfy.cli_args import args
from comfy_extras.nodes_video import SaveVideo as ComfySaveVideo
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
    return bool(re.search(r"(^|[^a-z0-9])(negative|neg)([^a-z0-9]|$)", title))


def _is_positive_clip_node(node):
    title = _node_title(node)
    return bool(re.search(r"(^|[^a-z0-9])(positive|pos)([^a-z0-9]|$)", title))


def _is_link_value(value):
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _normalize_prompt_graph(prompt):
    prompt = _json_loads_maybe(prompt)
    if not isinstance(prompt, dict):
        return None

    normalized_nodes = {}

    def collect_graph(graph):
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
            return

        link_sources = {}
        for link in graph.get("links", []):
            if isinstance(link, list) and len(link) >= 3:
                link_sources[str(link[0])] = [str(link[1]), int(link[2])]
            elif isinstance(link, dict):
                link_id = link.get("id")
                origin_id = link.get("origin_id")
                origin_slot = link.get("origin_slot", 0)
                if link_id is not None and origin_id is not None:
                    link_sources[str(link_id)] = [str(origin_id), int(origin_slot)]

        for node in graph.get("nodes", []):
            if not isinstance(node, dict) or node.get("id") is None:
                continue

            inputs = {}
            widget_index = 0
            widgets_values = node.get("widgets_values", []) or []
            for input_info in node.get("inputs", []) or []:
                if not isinstance(input_info, dict):
                    continue
                name = input_info.get("name")
                link = input_info.get("link")
                if name and link is not None and str(link) in link_sources:
                    inputs[str(name)] = link_sources[str(link)]
                elif name and input_info.get("widget") is not None and widget_index < len(widgets_values):
                    inputs[str(name)] = widgets_values[widget_index]
                    widget_index += 1

            normalized_nodes[str(node["id"])] = {
                "class_type": node.get("type") or node.get("class_type"),
                "_meta": {"title": node.get("title") or node.get("properties", {}).get("Node name for S&R", "")},
                "inputs": inputs,
                "widgets_values": widgets_values,
            }

    def walk(value):
        if isinstance(value, dict):
            collect_graph(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(prompt)

    if normalized_nodes:
        return normalized_nodes

    return prompt


def _node_input_link_ids(node, input_name):
    value = node.get("inputs", {}).get(input_name)
    if _is_link_value(value):
        return [str(value[0])]
    if isinstance(value, list):
        return [str(item[0]) for item in value if _is_link_value(item)]
    return []


def _first_widget_string(node):
    for value in node.get("widgets_values", []) or []:
        if isinstance(value, str) and value:
            return value
    return ""


def _resolve_text_input(prompt, value, visited):
    if isinstance(value, str):
        return value
    if _is_link_value(value):
        return _resolve_node_text(prompt, value[0], visited)
    return ""


def _resolve_node_text(prompt, node_id, visited=None):
    if visited is None:
        visited = set()

    node_id = str(node_id)
    if node_id in visited:
        return ""
    visited.add(node_id)

    node = prompt.get(node_id)
    if not isinstance(node, dict):
        return ""

    inputs = node.get("inputs", {})
    class_type = node.get("class_type")

    if class_type == "CLIPTextEncode":
        text = _resolve_text_input(prompt, inputs.get("text", ""), visited)
        return text or _first_widget_string(node)

    if class_type == "StringConcatenate":
        string_a = _resolve_text_input(prompt, inputs.get("string_a", ""), visited)
        string_b = _resolve_text_input(prompt, inputs.get("string_b", ""), visited)
        delimiter = _resolve_text_input(prompt, inputs.get("delimiter", ""), visited)
        if not string_a and not string_b:
            values = [value for value in (node.get("widgets_values", []) or []) if isinstance(value, str)]
            string_a = values[0] if len(values) > 0 else ""
            string_b = values[1] if len(values) > 1 else ""
            delimiter = values[2] if len(values) > 2 else delimiter
        if string_a and string_b:
            return f"{string_a}{delimiter}{string_b}"
        return string_a or string_b

    if isinstance(class_type, str) and class_type.startswith("TextGenerate"):
        return _resolve_text_input(prompt, inputs.get("prompt", ""), visited)

    for input_name in ("text", "prompt", "string", "string_a", "string_b", "value", "generated_text"):
        text = _resolve_text_input(prompt, inputs.get(input_name, ""), visited)
        if text:
            return text

    return _first_widget_string(node)


def _trace_clip_texts(prompt, node_id, visited=None):
    node_id = str(node_id)
    node = prompt.get(node_id)
    if not isinstance(node, dict):
        return []

    if node.get("class_type") == "CLIPTextEncode":
        text = _resolve_node_text(prompt, node_id, visited)
        return [text] if text else []

    if visited is None:
        visited = set()
    if node_id in visited:
        return []
    visited.add(node_id)

    texts = []
    for value in node.get("inputs", {}).values():
        if _is_link_value(value):
            texts.extend(_trace_clip_texts(prompt, value[0], visited))
    return texts


def _extract_clip_prompts_from_links(prompt):
    positive = ""
    negative = ""

    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        for node_id in _node_input_link_ids(node, "positive"):
            for text in _trace_clip_texts(prompt, node_id):
                if text and not positive:
                    positive = text

        for node_id in _node_input_link_ids(node, "negative"):
            for text in _trace_clip_texts(prompt, node_id):
                if text and not negative:
                    negative = text

        if positive and negative:
            return positive, negative

    return positive, negative


def _extract_clip_prompts(prompt):
    prompt = _normalize_prompt_graph(prompt)
    if not isinstance(prompt, dict):
        return "", ""

    positive, negative = _extract_clip_prompts_from_links(prompt)
    if positive or negative:
        return positive, negative

    clips = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        text = _resolve_node_text(prompt, node_id)
        if text:
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
    prompt = _normalize_prompt_graph(prompt)
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
        if value.startswith((b"Exif\x00\x00", b"MM\x00*", b"II*\x00")):
            return ""

        if value.startswith(b"UNICODE\x00"):
            value = value[len(b"UNICODE\x00"):]
            encodings = ("utf-16-be", "utf-16-le", "utf-8", "latin-1")
        elif value.startswith(b"ASCII\x00\x00\x00"):
            value = value[len(b"ASCII\x00\x00\x00"):]
            encodings = ("utf-8", "latin-1")
        elif value.startswith(b"JIS\x00\x00\x00\x00\x00"):
            value = value[len(b"JIS\x00\x00\x00\x00\x00"):]
            encodings = ("shift_jis", "utf-8", "latin-1")
        else:
            encodings = ("utf-8", "utf-16-be", "utf-16-le", "latin-1")

        for encoding in encodings:
            try:
                return value.decode(encoding).strip("\x00").strip()
            except Exception:
                pass
        return repr(value)
    return str(value)


def _decode_xp_exif_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip("\x00").strip()
    if isinstance(value, (tuple, list)):
        try:
            value = bytes(int(item) for item in value if int(item) >= 0)
        except Exception:
            return ""
    if isinstance(value, bytes):
        for encoding in ("utf-16-le", "utf-16-be", "utf-8", "latin-1"):
            try:
                return value.decode(encoding).strip("\x00").strip()
            except Exception:
                pass
    return ""


def _exif_tag_name(tag):
    try:
        return TAGS.get(int(tag), str(tag))
    except Exception:
        return str(tag)


def _load_exif_bytes(value):
    if not isinstance(value, bytes):
        return None

    for candidate in (value, value[6:] if value.startswith(b"Exif\x00\x00") else value):
        try:
            exif = Image.Exif()
            exif.load(candidate)
            if exif:
                return exif
        except Exception:
            pass
    return None


def _load_exif_text(value):
    if not isinstance(value, str):
        return None

    for encoding in ("utf-16-be", "utf-16-le", "latin-1"):
        try:
            exif = _load_exif_bytes(value.encode(encoding))
            if exif:
                return exif
        except Exception:
            pass
    return None


def _merge_exif_metadata(metadata, exif):
    if not exif:
        return

    xp_tags = {
        40091: "XPTitle",
        40092: "XPComment",
        40093: "XPAuthor",
        40094: "XPKeywords",
        40095: "XPSubject",
    }

    exif_map = {
        "ImageDescription": exif.get(Base.ImageDescription),
        "UserComment": exif.get(Base.UserComment),
        "Software": exif.get(Base.Software),
    }
    for key, value in exif_map.items():
        text = _decode_exif_text(value)
        if text and key not in metadata:
            metadata[key] = text

    all_items = []
    try:
        all_items.extend(list(exif.items()))
    except Exception:
        pass

    for ifd in (IFD.Exif, IFD.GPSInfo, IFD.Interop, IFD.IFD1):
        try:
            all_items.extend(list(exif.get_ifd(ifd).items()))
        except Exception:
            pass

    for tag, value in all_items:
        key = xp_tags.get(int(tag), _exif_tag_name(tag))
        if key in metadata:
            continue
        if int(tag) in xp_tags:
            text = _decode_xp_exif_text(value)
        elif isinstance(value, (str, bytes)):
            text = _decode_exif_text(value)
        elif isinstance(value, (tuple, list)) and value and all(isinstance(item, int) for item in value):
            text = _decode_xp_exif_text(value)
        else:
            text = ""
        if text:
            metadata[key] = text


def _read_image_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = {}

        for key, value in img.info.items():
            if isinstance(value, bytes) and str(key).lower() == "exif":
                _merge_exif_metadata(metadata, _load_exif_bytes(value))
                metadata[str(key)] = f"<binary EXIF: {len(value)} bytes>"
            elif isinstance(value, bytes):
                metadata[str(key)] = _decode_exif_text(value)
            elif isinstance(value, str) and str(key).lower() == "exif":
                _merge_exif_metadata(metadata, _load_exif_text(value))
                metadata[str(key)] = "<binary EXIF text>"
            elif isinstance(value, str):
                metadata[str(key)] = value

        try:
            exif = img.getexif()
        except Exception:
            exif = None

        if exif:
            _merge_exif_metadata(metadata, exif)

        # JPEG output from DTVSaveImageWithMeta stores the same key/value
        # metadata as PNG in ImageDescription, wrapped as JSON.
        description = _json_loads_maybe(metadata.get("ImageDescription"))
        if isinstance(description, dict) and description.get("dtv_metadata_version") == 1:
            saved_metadata = description.get("metadata")
            if isinstance(saved_metadata, dict):
                metadata.update(saved_metadata)

        if "parameters" not in metadata:
            preferred_keys = (
                "parameters",
                "UserComment",
                "ImageDescription",
                "XPComment",
                "XPSubject",
                "XPTitle",
                "comment",
                "Comment",
                "Description",
            )
            values = [metadata.get(key) for key in preferred_keys]
            values.extend(value for key, value in metadata.items() if key not in preferred_keys)
            for value in values:
                if value and ("Negative prompt:" in value or "Steps:" in value):
                    metadata["parameters"] = value
                    break

        return metadata


def _annotated_image_name(filename, subfolder=None, folder_type=None):
    if not filename:
        return ""

    filename = str(filename)
    if subfolder:
        filename = os.path.join(str(subfolder), filename)

    if folder_type in {"input", "output", "temp"} and not re.search(r"\s\[(input|output|temp)\]$", filename):
        filename = f"{filename} [{folder_type}]"

    return filename


def _extract_prompts_from_metadata(metadata):
    parameters = metadata.get("parameters", "")
    positive, negative = _parse_parameters(parameters)

    if not positive and not negative:
        positive, negative = _extract_clip_prompts(metadata.get("prompt"))
        if not positive and not negative:
            positive, negative = _extract_clip_prompts(metadata.get("workflow"))
        if not parameters:
            parameters = _build_parameters(metadata.get("prompt")) or _build_parameters(metadata.get("workflow"))

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


class DTVSaveVideo(ComfySaveVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "DTVSaveVideo"
        schema.search_aliases = ["save video", "export video", "dtv save video"]
        schema.display_name = "DTV Save Video"
        schema.category = "DTV Restore Prompts"
        schema.essentials_category = None
        schema.description = (
            "Saves the input videos to your ComfyUI output directory and allows "
            "the preview component to be resized smaller."
        )
        return schema

    @classmethod
    def execute(cls, video, filename_prefix, *args, **kwargs):
        filename_prefix = _expand_date_tokens(filename_prefix)
        return super().execute(video, filename_prefix, *args, **kwargs)


@PromptServer.instance.routes.get("/dtv_restore_prompts/read_metadata")
async def read_metadata_route(request):
    image = request.rel_url.query.get("image")
    subfolder = request.rel_url.query.get("subfolder")
    folder_type = request.rel_url.query.get("type")
    if not image:
        return web.json_response({"error": "Missing image"}, status=400)

    try:
        image_path = folder_paths.get_annotated_filepath(_annotated_image_name(image, subfolder, folder_type))
        parsed = _extract_prompts_from_metadata(_read_image_metadata(image_path))
        return web.json_response(parsed)
    except FileNotFoundError as err:
        return web.json_response({"error": f"File not found: {err.filename}"}, status=404)
    except Exception as err:
        return web.json_response({"error": str(err)}, status=500)


NODE_CLASS_MAPPINGS = {
    "DTVSaveImageWithMeta": DTVSaveImageWithMeta,
    "DTVReadPromptsFromPNGMetadata": DTVReadPromptsFromPNGMetadata,
    "DTVSaveVideo": DTVSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DTVSaveImageWithMeta": "DTV Save Image (Save meta)",
    "DTVReadPromptsFromPNGMetadata": "DTV Read Prompts From PNG Metadata",
    "DTVSaveVideo": "DTV Save Video",
}
