# comfyui_dtv_pack

Custom nodes for ComfyUI focused on saving images with prompt metadata and restoring prompts from image metadata.

## Nodes

### DTV Save Image (Save meta)

Saves generated images while preserving prompt metadata.

- Save as PNG or JPEG
- JPEG quality setting from `0` to `100`
- Default JPEG quality: `85`
- Keeps ComfyUI prompt metadata when metadata saving is enabled
- Supports date tokens in `filename_prefix`, for example `%date:yyyyMMdd_HHmmss%`

### DTV Read Prompts From PNG Metadata

Reads prompt metadata from an uploaded image and outputs:

- `positive`
- `negative`
- `parameters`
- `raw_metadata`

The `parameters` field is shown as a normal multiline text area in the node UI.

> Despite the node name, metadata saved by `DTV Save Image (Save meta)` can also be restored from JPEG files created by this pack.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/elledartois/comfyui_dtv_pack.git
```

Then restart ComfyUI.

## Requirements

This node pack uses libraries already included with a normal ComfyUI installation:

- Pillow
- NumPy
- aiohttp

## Notes

- PNG metadata is stored using PNG text chunks.
- JPEG metadata is stored in EXIF. The metadata payload is wrapped so it can be read back by this node pack.
- ComfyUI's global metadata setting is respected. If metadata is disabled globally, this pack will not write prompt metadata.
