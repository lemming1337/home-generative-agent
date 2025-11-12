# Frontend Image Upload Implementation Instructions

The backend has been updated to support image uploads, but the frontend card needs to be modified in the source repository.

## Source Repository
https://github.com/lemming1337/homeassistant-assist-card

## Required Frontend Changes

### 1. Add File Input to Card UI

Add a file input button to the card template:

```html
<input
  type="file"
  accept="image/jpeg,image/png,image/webp,image/gif"
  @change="${this._handleImageUpload}"
  id="imageInput"
  style="display: none;"
/>
<button @click="${this._triggerImageUpload}">
  📷 Upload Image
</button>
```

### 2. Add Image Preview Area

```html
<div class="image-preview" ?hidden="${!this._uploadedImages.length}">
  ${this._uploadedImages.map((img, idx) => html`
    <div class="image-preview-item">
      <img src="${img.dataUrl}" />
      <button @click="${() => this._removeImage(idx)}">✕</button>
    </div>
  `)}
</div>
```

### 3. Add Image Handling Methods

```javascript
_triggerImageUpload() {
  this.shadowRoot.querySelector('#imageInput').click();
}

async _handleImageUpload(e) {
  const files = e.target.files;
  for (const file of files) {
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      alert('Image too large. Maximum size is 10MB.');
      continue;
    }

    const dataUrl = await this._readFileAsDataURL(file);
    this._uploadedImages.push({
      file,
      dataUrl
    });
  }
  this.requestUpdate();
}

_readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

_removeImage(index) {
  this._uploadedImages.splice(index, 1);
  this.requestUpdate();
}
```

### 4. Modify Message Sending

Update the `_sendMessage` method to include images in the expected format:

```javascript
async _sendMessage() {
  let messageText = this._inputText;

  // Append images in the format: [IMAGE:data:image/...;base64,...]
  for (const img of this._uploadedImages) {
    messageText = `[IMAGE:${img.dataUrl}]` + messageText;
  }

  const response = await this.hass.callWS({
    type: "conversation/process",
    text: messageText,
    conversation_id: this._conversationId,
    pipeline_id: this._config?.pipeline_id
  });

  // Clear uploaded images after sending
  this._uploadedImages = [];
  this._inputText = '';
  this.requestUpdate();
}
```

### 5. Add CSS Styles

```css
.image-preview {
  display: flex;
  gap: 8px;
  padding: 8px;
  flex-wrap: wrap;
}

.image-preview-item {
  position: relative;
  width: 100px;
  height: 100px;
}

.image-preview-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 4px;
}

.image-preview-item button {
  position: absolute;
  top: -8px;
  right: -8px;
  background: red;
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  cursor: pointer;
}
```

### 6. Initialize State

In the constructor, add:

```javascript
constructor() {
  super();
  this._uploadedImages = [];
}
```

## Testing

1. Build the frontend card in the source repository
2. Copy the built file to `custom_components/home_generative_agent/www/homeassistant-assist-card.js`
3. Clear browser cache and reload Home Assistant
4. Test image upload with the card

## Image Format

Images are sent to the backend in the format:
```
[IMAGE:data:image/jpeg;base64,/9j/4AAQSkZJRg...]User message text here
```

The backend will extract the images and process them with the VLM before the normal conversation flow.
