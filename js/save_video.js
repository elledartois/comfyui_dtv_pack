import { app } from "../../scripts/app.js";

const DTV_SAVE_VIDEO_NODE = "DTVSaveVideo";
const MIN_NODE_WIDTH = 220;
const MIN_PREVIEW_HEIGHT = 120;

function getWidgetAreaHeight(node) {
  const widgetRowHeight = globalThis.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20;
  return (node.widgets?.length ?? 0) * widgetRowHeight + 80;
}

function hasVideoPreview(node) {
  return Boolean(
    Array.isArray(node.imgs) ||
      Array.isArray(node.images) ||
      Array.isArray(node.ui?.images) ||
      Array.isArray(node.widgets?.find((widget) => widget.name === "images")?.value)
  );
}

function patchNodeComputeSize(node) {
  if (node._dtvSaveVideoComputeSizePatched || typeof node.computeSize !== "function") {
    return;
  }

  const originalComputeSize = node.computeSize.bind(node);
  node.computeSize = function () {
    const size = originalComputeSize(...arguments);
    if (!Array.isArray(size) || !hasVideoPreview(this)) {
      return size;
    }

    const currentWidth = this.size?.[0] ?? size[0];
    const currentHeight = this.size?.[1] ?? size[1];
    const minHeight = Math.max(MIN_PREVIEW_HEIGHT, getWidgetAreaHeight(this));

    return [
      Math.min(size[0], Math.max(MIN_NODE_WIDTH, currentWidth)),
      Math.min(size[1], Math.max(minHeight, currentHeight)),
    ];
  };

  node._dtvSaveVideoComputeSizePatched = true;
}

function patchPreviewWidgetSizing(node) {
  for (const widget of node.widgets ?? []) {
    if (widget._dtvSaveVideoSizingPatched || typeof widget.computeSize !== "function") {
      continue;
    }

    const looksLikePreview =
      widget.name === "image" ||
      widget.name === "images" ||
      widget.type === "image" ||
      widget.type === "preview" ||
      Array.isArray(widget.value?.images) ||
      Array.isArray(widget.options?.images);

    if (!looksLikePreview) {
      continue;
    }

    const originalComputeSize = widget.computeSize.bind(widget);
    widget.computeSize = function (width) {
      const size = originalComputeSize(width);
      if (!Array.isArray(size)) {
        return size;
      }

      return [
        Math.min(size[0], Math.max(MIN_NODE_WIDTH, node.size?.[0] ?? width)),
        Math.min(size[1], Math.max(MIN_PREVIEW_HEIGHT, node.size?.[1] ?? size[1])),
      ];
    };

    widget._dtvSaveVideoSizingPatched = true;
  }
}

function allowVideoPreviewShrink(node) {
  node.resizable = true;
  node.min_size = [MIN_NODE_WIDTH, MIN_PREVIEW_HEIGHT];
  node._min_size = [MIN_NODE_WIDTH, MIN_PREVIEW_HEIGHT];
  patchNodeComputeSize(node);
  patchPreviewWidgetSizing(node);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "dtv.save.video",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== DTV_SAVE_VIDEO_NODE) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      requestAnimationFrame(() => {
        allowVideoPreviewShrink(this);
      });
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function () {
      const result = onExecuted?.apply(this, arguments);
      requestAnimationFrame(() => {
        allowVideoPreviewShrink(this);
      });
      return result;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      const result = onResize?.apply(this, arguments);
      allowVideoPreviewShrink(this);
      return result;
    };
  },
});
