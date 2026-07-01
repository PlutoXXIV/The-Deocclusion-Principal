# deeplab_model_on.py
import os
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import torchvision.transforms as T

# ---- Global model cache ----
_MODEL = None
_DEVICE = None

# Try to get model path from environment, otherwise use a relative path
# Assume model is in the project root (one level up from scripts/)
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessed_model_2.pth')
_MODEL_PATH = os.environ.get('MODEL_PATH', _DEFAULT_MODEL_PATH)

IMG_SIZE = 1024
THRESHOLD = 0.5

class FixedImageEnhancer:
    @staticmethod
    def enhance(img_bgr):
        img = img_bgr.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        img = cv2.convertScaleAbs(img, alpha=1.1, beta=0)
        gamma = 0.95
        lut = np.array([min(255, int(255 * (i/255) ** (1.0/gamma))) for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)
        return img.astype(np.uint8)

def load_image(path):
    img = cv2.imread(path)                      # BGR
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = FixedImageEnhancer.enhance(img)       # enhance in BGR
    return img                                  # still BGR

def to_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = normalize(t)
    return t.unsqueeze(0).to(_DEVICE)

def load_model(model_path=None):
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL
    if model_path is None:
        model_path = _MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} on {_DEVICE}")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(_DEVICE)
    ckpt = torch.load(model_path, map_location=_DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return model

@torch.no_grad()
def run_inference(img_bgr):
    model = load_model()
    x = to_tensor(img_bgr)
    logits = model(x)
    prob = torch.sigmoid(logits)
    mask = (prob.squeeze().cpu().numpy() > THRESHOLD).astype(np.uint8)
    return mask

def predict_mask(input_path, output_path):
    """Main entry point: generates a binary mask from a satellite image."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    img = load_image(input_path)
    mask = run_inference(img)
    cv2.imwrite(output_path, mask * 255)
    print(f"Mask saved to {output_path}")

# ---- If run as script, test on a sample image ----
if __name__ == "__main__":
    import glob
    # Use relative test directory: io/ in the project root
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io')
    test_files = sorted(glob.glob(os.path.join(test_dir, '*_sat.jpg')))
    if not test_files:
        test_files = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))
    if test_files:
        sample = test_files[0]
        out = "test_mask.png"
        predict_mask(sample, out)
        print(f"Test completed. Mask saved to {out}")
    else:
        print(f"No test images found in {test_dir}")