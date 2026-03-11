import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import base64

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
    """U-Net for oil spill detection."""

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


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = LegacyUNet().to(device)
    checkpoint = torch.load(
        "best_model_1 (1).pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def predict(image_pil, model, device, threshold=0.5):
    image_pil = image_pil.convert("L").resize((512, 512))
    arr = np.asarray(image_pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

    pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
    prob_np = probs.squeeze().cpu().numpy()

    oil_pixels = int(pred_np.sum())
    total_pixels = int(pred_np.size)
    oil_ratio = oil_pixels / total_pixels

    mask_img = Image.fromarray((pred_np * 255).astype(np.uint8), mode="L")

    return {
        "oil_ratio": oil_ratio,
        "oil_pixels": oil_pixels,
        "total_pixels": total_pixels,
        "avg_probability": float(prob_np.mean()),
        "max_probability": float(prob_np.max()),
        "mask": mask_img,
    }


st.set_page_config(page_title="Marine Oil Spill Detection", layout="wide")

st.title("🌊 Marine Oil Spill Detection")
st.markdown(
    "Upload a SAR (Synthetic Aperture Radar) image to detect oil spills."
)

model, device = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a SAR image (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Settings")
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_file:
    if st.button("Run Prediction", key="predict_btn"):
        with st.spinner("Processing..."):
            result = predict(image, model, device, threshold)

        st.success("Prediction complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Results")
            st.metric("Oil Ratio", f"{result['oil_ratio']:.4f}")
            st.metric("Oil Pixels", f"{result['oil_pixels']:,}")
            st.metric("Total Pixels", f"{result['total_pixels']:,}")
            st.metric("Avg Probability", f"{result['avg_probability']:.4f}")
            st.metric("Max Probability", f"{result['max_probability']:.4f}")

        with col2:
            st.subheader("Predicted Mask")
            st.image(
                result["mask"],
                caption="Oil Spill Mask (White=Oil, Black=Water)",
                use_column_width=True,
            )

        st.download_button(
            label="Download Mask",
            data=result["mask"].tobytes(),
            file_name="oil_spill_mask.png",
            mime="image/png",
        )

st.markdown("---")
st.markdown(
    "**Model**: U-Net | **Dataset**: SAR Oil Spill Images | **Framework**: PyTorch"
)
