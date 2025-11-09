import torch
from torchvision import transforms
from PIL import Image
from timm.models.swin_transformer import SwinTransformer
import numpy as np

# -------------------
# Configuration
# -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'model_repo/weights/swin_nih_chestxray14.pth'

IMAGE_PATH = "test.png"  # change this if you want to test another image

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

# -------------------
# Load Model
# -------------------
def load_model(model_path):
    print("🔄 Loading Swin Transformer model (NIH pretrained)...")
    import torch.serialization
    torch.serialization.add_safe_globals([SwinTransformer])

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = SwinTransformer(img_size=224, num_classes=len(LABELS))
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
    return model

# -------------------
# Preprocess Image
# -------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)

# -------------------
# Predict with Improved Temperature Scaling
# -------------------
@torch.no_grad()
def predict(model, img_tensor):
    outputs = model(img_tensor)
    if isinstance(outputs, list):
        outputs = outputs[0]

    temperature = 0.4  # or even 0.5 for stronger confidences
    scaled_outputs = outputs / temperature

    probs = torch.sigmoid(scaled_outputs)[0]

    # remove normalization for independent confidences
    results = sorted(zip(LABELS, probs.tolist()), key=lambda x: x[1], reverse=True)
    return results


# -------------------
# Main Execution
# -------------------
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    img_tensor = preprocess_image(IMAGE_PATH)

    results = predict(model, img_tensor)

    print("\n🩻 Top 5 Predictions:")
    for label, prob in results[:5]:
        print(f"{label}: {prob * 100:.2f}%")

    # ------------------- Filter by confidence threshold -------------------
threshold = 0.6  # set your threshold here (e.g., 0.6 = 60%)
print(f"\n🩺 Diseases with confidence > {threshold * 100:.0f}%:")

# for label, prob in results:
#     if prob > threshold:
#         print(f"{label}: {prob * 100:.2f}%")

