import { app } from "../../scripts/app.js";

function createTextareaWidget(node, name) {
  const widget = node.widgets?.find((widget) => widget.name === name);
  if (widget) {
    widget.serialize = false;
  }
  return widget;
}

function setWidgetValue(widget, value) {
  const text = value ?? "";
  if (!widget) {
    return;
  }
  widget.value = text;
  if (widget.inputEl) {
    widget.inputEl.value = text;
  }
}

app.registerExtension({
  name: "dtv.restore.prompts",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "DTVReadPromptsFromPNGMetadata") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this._dtvTextWidgets = {
        parameters: createTextareaWidget(this, "parameters"),
      };
      for (const widget of Object.values(this._dtvTextWidgets)) {
        if (widget) {
          widget.serialize = false;
        }
      }
      requestAnimationFrame(() => {
        refreshMetadata.call(this);
      });
      this._dtvImageWatchTimer = window.setInterval(() => {
        refreshMetadataIfImageChanged.call(this);
      }, 500);
      return result;
    };

    const getImageValue = function () {
      const imageWidget = this.widgets?.find((widget) => widget.name === "image");
      return imageWidget?.value ?? "";
    };

    const getImageInfo = function () {
      const value = getImageValue.call(this);
      if (!value) {
        return null;
      }

      if (typeof value === "string") {
        return { image: value, key: value };
      }

      if (typeof value === "object") {
        const image = value.filename ?? value.name ?? value.image ?? "";
        if (!image) {
          return null;
        }
        const subfolder = value.subfolder ?? "";
        const type = value.type ?? "input";
        return {
          image,
          subfolder,
          type,
          key: JSON.stringify({ image, subfolder, type }),
        };
      }

      const image = String(value);
      return { image, key: image };
    };

    const refreshMetadata = async function (attempt = 0) {
      if (!this._dtvTextWidgets) {
        return;
      }

      const imageInfo = getImageInfo.call(this);
      if (!imageInfo) {
        return;
      }
      this._dtvLastRefreshImage = imageInfo.key;
      this._dtvRefreshToken = (this._dtvRefreshToken ?? 0) + 1;
      const refreshToken = this._dtvRefreshToken;

      try {
        const query = new URLSearchParams();
        query.set("image", imageInfo.image);
        if (imageInfo.subfolder) {
          query.set("subfolder", imageInfo.subfolder);
        }
        if (imageInfo.type) {
          query.set("type", imageInfo.type);
        }

        const response = await fetch(`/dtv_restore_prompts/read_metadata?${query.toString()}`);
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.error ?? "Failed to read metadata");
        }

        if (refreshToken !== this._dtvRefreshToken) {
          return;
        }
        setWidgetValue(this._dtvTextWidgets.parameters, data.parameters);
        this.setDirtyCanvas(true, true);
      } catch (error) {
        const fileMayStillBeUploading =
          error.message.includes("No such file or directory") ||
          error.message.includes("File not found") ||
          error.message.includes("Invalid file path") ||
          error.message.includes("404") ||
          error.message.includes("Failed to fetch");

        if (fileMayStillBeUploading && attempt < 20) {
          window.setTimeout(() => {
            if (imageInfo.key === getImageInfo.call(this)?.key) {
              refreshMetadata.call(this, attempt + 1);
            }
          }, 500);
          return;
        }

        if (refreshToken !== this._dtvRefreshToken) {
          return;
        }
        setWidgetValue(this._dtvTextWidgets.parameters, `Metadata read failed: ${error.message}`);
        this.setDirtyCanvas(true, true);
      }
    };

    const refreshMetadataIfImageChanged = function () {
      const imageInfo = getImageInfo.call(this);
      if (imageInfo && imageInfo.key !== this._dtvLastSeenImage) {
        this._dtvLastSeenImage = imageInfo.key;
        refreshMetadata.call(this);
      }
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (!this._dtvTextWidgets || !message) {
        return;
      }
      setWidgetValue(this._dtvTextWidgets.parameters, message.parameters?.[0]);
      this.setDirtyCanvas(true, true);
    };

    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (name, value, oldValue, widget) {
      const result = onWidgetChanged?.apply(this, arguments);
      const oldKey = typeof oldValue === "object" ? JSON.stringify(oldValue) : oldValue;
      const newKey = typeof value === "object" ? JSON.stringify(value) : value;
      if (name === "image" && newKey !== oldKey) {
        this._dtvLastSeenImage = newKey;
        refreshMetadata.call(this);
      }
      return result;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
      onDrawForeground?.apply(this, arguments);
      refreshMetadataIfImageChanged.call(this);
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._dtvImageWatchTimer) {
        window.clearInterval(this._dtvImageWatchTimer);
        this._dtvImageWatchTimer = null;
      }
      return onRemoved?.apply(this, arguments);
    };
  },
});
