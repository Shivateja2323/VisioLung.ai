import torch
import timm
from PIL import Image
from torchvision import transforms

# ✅ Load your uploaded image
img_path = "test.png"  # Use your uploaded image name
img = Image.open(img_path).convert("RGB")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(img).unsqueeze(0)

# Load pretrained Swin Transformer
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
model.eval()

# Inference
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5 = torch.topk(probs, 5)

# Load ImageNet labels
import requests
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(labels_url).json()

print("\n🧠 Top 5 Predictions:")
for idx, (val, prob) in enumerate(zip(top5.indices, top5.values)):
    print(f"{idx+1}. {labels[val]} ({prob.item()*100:.2f}%)")
