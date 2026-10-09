import io, os, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from skimage.filters import threshold_otsu
from huggingface_hub import hf_hub_download

# Tối ưu PyTorch chỉ chạy 1 thread để tránh tốn RAM/CPU trên Render
torch.set_num_threads(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.strided_conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.strided_conv_block(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)
        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

device = torch.device("cpu")
model = UNet(n_channels=2, n_classes=1).to(device)

MODEL_REPO = "nhkv10905/unet-aapm-non-metal"
weights_path = hf_hub_download(repo_id=MODEL_REPO, filename="unet_best.pth")
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

HU_SCALE = 1000.0

def create_metal_mask_2d(recon_hu, body_threshold_hu=150.0):
    recon_hu = np.asarray(recon_hu, dtype=np.float32)
    finite = np.isfinite(recon_hu)
    body_mask = finite & (recon_hu >= body_threshold_hu)
    body_values = recon_hu[body_mask]
    if body_values.size < 32:
        return np.zeros_like(recon_hu, dtype=np.float32)
    p95_hu = np.percentile(body_values, 95)
    tail = body_values[body_values >= p95_hu]
    if tail.size < 2: otsu_hu = p95_hu
    elif np.all(tail == tail[0]): otsu_hu = float(tail[0])
    else: otsu_hu = float(threshold_otsu(tail))
    threshold_hu = max(p95_hu, otsu_hu)
    return (body_mask & (recon_hu >= threshold_hu)).astype(np.float32)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return Response(status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Kiểm tra dung lượng file
        if len(contents) != 512 * 512 * 4: # 1,048,576 bytes
            raise HTTPException(status_code=400, detail="File size must be exactly 1MB (512x512 float32)")
            
        artifact_raw = np.frombuffer(contents, dtype=np.float32).reshape((512, 512)).copy()
        
        mask = create_metal_mask_2d(artifact_raw)
        artifact = np.clip(artifact_raw, -1024, 3071).astype(np.float32)
        
        art_n = torch.from_numpy((artifact/HU_SCALE)[None, None]).float().to(device)
        mask_t = torch.from_numpy(mask[None, None]).float().to(device)
        art_hu_t = torch.from_numpy(artifact[None, None]).float().to(device)
        
        model_input = torch.cat([art_n, mask_t], dim=1)
        
        with torch.no_grad():
            pred_n = model(model_input)
            pred_hu = pred_n * HU_SCALE
            metal_bool = mask_t > 0.5
            pred_hu[metal_bool] = art_hu_t[metal_bool]
            
        pred_np = pred_hu.numpy()[0, 0].astype(np.float32)
        return Response(content=pred_np.tobytes(), media_type="application/octet-stream")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))