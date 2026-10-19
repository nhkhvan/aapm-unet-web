import modal

# 1. Định nghĩa môi trường (Image)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "numpy",
        "scikit-image",
        "huggingface_hub",
        "fastapi[standard]",
        "python-multipart"
    )
)

app = modal.App("aapm-unet-backend")

# 2. Class Model nạp weights U-Net chạy trên CPU miễn phí
@app.cls(
    image=image,
    cpu=2.0, # Dùng CPU không cần liên kết thẻ VISA/Mastercard
    scaledown_window=300
)
class Model:
    @modal.enter()
    def load_weights(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from huggingface_hub import hf_hub_download

        class DoubleConv(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
                )
            def forward(self, x): return self.conv(x)

        class Down(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, in_c, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
                    DoubleConv(in_c, out_c)
                )
            def forward(self, x): return self.conv(x)

        class Up(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = DoubleConv(in_c, out_c)
            def forward(self, x1, x2):
                x1 = self.up(x1)
                diffY, diffX = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
                x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
                return self.conv(torch.cat([x2, x1], dim=1))

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.inc = DoubleConv(2, 64)
                self.down1 = Down(64, 128)
                self.down2 = Down(128, 256)
                self.down3 = Down(256, 512)
                self.down4 = Down(512, 512)
                self.up1 = Up(1024, 256)
                self.up2 = Up(512, 128)
                self.up3 = Up(256, 64)
                self.up4 = Up(128, 64)
                self.outc = nn.Conv2d(64, 1, 1)
            def forward(self, x):
                x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
                x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
                return self.outc(x)

        self.device = torch.device("cpu") # Chạy bằng CPU
        self.model = UNet().to(self.device)
        weights_path = hf_hub_download(repo_id="nhkv10905/unet-aapm-non-metal", filename="unet_best.pth")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    @modal.method()
    def predict_image(self, artifact_bytes: bytes, gt_bytes: bytes = None):
        import numpy as np
        import torch
        from skimage.filters import threshold_otsu

        artifact_raw = np.frombuffer(artifact_bytes, dtype=np.float32).reshape((512, 512)).copy()
        
        recon_hu = np.asarray(artifact_raw, dtype=np.float32)
        body_mask = np.isfinite(recon_hu) & (recon_hu >= 150.0)
        body_values = recon_hu[body_mask]
        if body_values.size < 32:
            mask = np.zeros_like(recon_hu, dtype=np.float32)
        else:
            p95_hu = np.percentile(body_values, 95)
            tail = body_values[body_values >= p95_hu]
            otsu_hu = float(threshold_otsu(tail)) if tail.size >= 2 else p95_hu
            mask = (body_mask & (recon_hu >= max(p95_hu, otsu_hu))).astype(np.float32)

        artifact = np.clip(artifact_raw, -1024, 3071).astype(np.float32)
        HU_SCALE = 1000.0

        art_n = torch.from_numpy((artifact / HU_SCALE)[None, None]).float().to(self.device)
        mask_t = torch.from_numpy(mask[None, None]).float().to(self.device)
        art_hu_t = torch.from_numpy(artifact[None, None]).float().to(self.device)
        model_input = torch.cat([art_n, mask_t], dim=1)

        with torch.no_grad():
            pred_n = self.model(model_input)
            pred_hu = pred_n * HU_SCALE
            metal_bool = mask_t > 0.5
            pred_hu = pred_hu.clone()
            pred_hu[metal_bool] = art_hu_t[metal_bool]

        pred_np = pred_hu.cpu().numpy()[0, 0].astype(np.float32)

        has_gt = False
        rmse_total, rmse_nonmetal, aapm_score = 0.0, 0.0, 0.0

        if gt_bytes and len(gt_bytes) == 512 * 512 * 4:
            gt_raw = np.frombuffer(gt_bytes, dtype=np.float32).reshape((512, 512)).copy()
            gt_clean = np.clip(gt_raw, -1024, 3071).astype(np.float32)

            non_metal_mask = (mask < 0.5)
            rmse_total = float(np.sqrt(np.mean((gt_clean - pred_np) ** 2)))
            rmse_nonmetal = float(np.sqrt(np.mean((gt_clean[non_metal_mask] - pred_np[non_metal_mask]) ** 2))) if np.any(non_metal_mask) else rmse_total
            aapm_score = max(0.0, round(100.0 - (rmse_nonmetal / 2.0), 1))
            has_gt = True

        return pred_np.tobytes(), has_gt, rmse_total, rmse_nonmetal, aapm_score

# 3. Web Endpoint FastAPI
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def predict(request: modal.Request):
    from fastapi.responses import Response

    form = await request.form()
    file_item = form.get("file")
    gt_item = form.get("gt_file")

    if not file_item:
        return Response(content="Thiếu file đầu vào", status_code=400)

    artifact_bytes = await file_item.read()
    gt_bytes = await gt_item.read() if gt_item and hasattr(gt_item, "read") else None

    model = Model()
    pred_bytes, has_gt, rmse_total, rmse_nonmetal, aapm_score = model.predict_image.remote(
        artifact_bytes, gt_bytes
    )

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Expose-Headers": "X-HAS-GT, X-RMSE-TOTAL, X-RMSE-NONMETAL, X-AAPM-SCORE",
        "X-HAS-GT": "TRUE" if has_gt else "FALSE"
    }

    if has_gt:
        headers["X-RMSE-TOTAL"] = f"{rmse_total:.2f} HU"
        headers["X-RMSE-NONMETAL"] = f"{rmse_nonmetal:.2f} HU"
        headers["X-AAPM-SCORE"] = f"{aapm_score}/100"

    return Response(content=pred_bytes, media_type="application/octet-stream", headers=headers)