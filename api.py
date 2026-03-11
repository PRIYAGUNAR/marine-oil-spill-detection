import base64
import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LegacyUNet(nn.Module):
    """Architecture that matches best_model_1 (1).pth."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.down1 = DoubleConv(1, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        b = self.bottleneck(self.pool(x3))

        u3 = self.up3(b)
        c3 = self.conv3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, x1], dim=1))

        return self.final(c1)


CHECKPOINT_PATH = "best_model_1 (1).pth"
IMAGE_SIZE = 512
app = FastAPI(title="Marine Oil Spill Detection API", version="1.0.0")


def load_model(checkpoint_path: str) -> LegacyUNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LegacyUNet().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def mask_to_base64_png(mask: np.ndarray) -> str:
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.on_event("startup")
def startup_event() -> None:
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "device": str(device), "checkpoint": CHECKPOINT_PATH})


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = 0.5, return_mask: Optional[bool] = True) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    if threshold is None or threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    image_bytes = await file.read()
    x = preprocess_image(image_bytes).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
    prob_np = probs.squeeze().cpu().numpy()

    oil_pixels = int(pred_np.sum())
    total_pixels = int(pred_np.size)
    oil_ratio = oil_pixels / total_pixels

    payload = {
        "threshold": threshold,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "oil_pixels": oil_pixels,
        "total_pixels": total_pixels,
        "oil_ratio": oil_ratio,
        "avg_probability": float(prob_np.mean()),
        "max_probability": float(prob_np.max()),
    }

    if return_mask:
        payload["mask_png_base64"] = mask_to_base64_png(pred_np)

    return JSONResponse(payload)
