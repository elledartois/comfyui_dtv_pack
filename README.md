# comfyui_dtv_pack

標準のSave Imageノードに､Stable Diffusion WebUI A1111のように｢生成画像に生成時のプロンプトをメタデータで保存する｣機能を追加しました｡
また､すでにメタデータを持った画像に保存されているメタデータテキストを確認する機能を作成しました｡

Custom nodes for ComfyUI focused on saving images with prompt metadata and restoring prompts from image metadata.

## Nodes

### DTV Save Image (Save meta)

Save Imageノードの代わりに利用します｡PNGだけでなくJPEGでの保存もできるようになっています｡

Saves generated images while preserving prompt metadata.

- Save as PNG or JPEG
- JPEG quality setting from `0` to `100`
- Default JPEG quality: `85`
- Keeps ComfyUI prompt metadata when metadata saving is enabled
- Supports date tokens in `filename_prefix`, for example `%date:yyyyMMdd_HHmmss%`

### DTV Save Video

標準のSave Videoノードをベースにした動画保存ノードです｡標準ノードでは動画プレビュー部分が実サイズに引っ張られて小さくできない場合がありますが､このノードではコンポーネントの縮小表示を許可しています｡

Video save node based on ComfyUI's standard Save Video node.

- Saves input videos to the ComfyUI output directory
- Supports the same format / codec options as the standard Save Video node
- Supports date tokens in `filename_prefix`, for example `%date:yyyyMMdd_HHmmss%`
- Allows the video preview component to be resized smaller than the actual video size

### DTV Read Prompts From PNG Metadata

"DTV Save Image"ノードで保存した画像にあるメタデータテキストを表示させるだけのノードです｡

Reads prompt metadata from an uploaded image and outputs:

- `positive`
- `negative`
- `parameters`
- `raw_metadata`

The `parameters` field is shown as a normal multiline text area in the node UI.

> Despite the node name, metadata saved by `DTV Save Image (Save meta)` can also be restored from JPEG files created by this pack.

## Installation

custom_nodesフォルダにgirt cloneコマンドでチェックアウトしてください｡

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
