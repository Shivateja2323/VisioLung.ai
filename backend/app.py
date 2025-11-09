from flask import Flask, request, jsonify, send_file
import sys, os
import torch
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader
import random
import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import io
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import json
import os
from datetime import datetime, timedelta
from flask import Flask, jsonify, request




STATS_FILE = "scan_history.json"

def load_scans():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return []

def save_scans(data):
    with open(STATS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ============================================================
# 1️⃣ Flask Setup
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# 2️⃣ Fix path and import model
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "model_repo"))
from inference_nih import load_model, preprocess_image, predict

MODEL_PATH = os.path.join(BASE_DIR, "model_repo", "weights", "swin_nih_chestxray14.pth")

print("🔄 Loading Swin Transformer model...")
model = load_model(MODEL_PATH)
print("✅ Swin Transformer loaded successfully!")

# ============================================================
# 7️⃣ dashboard stat
# ============================================================

@app.route("/dashboard_stats", methods=["GET"])
def dashboard_stats():
    scans = load_scans()
    if not scans:
        return jsonify({
            "total_today": 0,
            "avg_confidence": 0,
            "most_common": "N/A",
            "trend": [],
            "all_scans": []
        })

    # ✅ Convert timestamps to datetime safely
    for scan in scans:
        if isinstance(scan["timestamp"], str):
            try:
                scan["timestamp"] = datetime.fromisoformat(scan["timestamp"])
            except ValueError:
                continue

    now = datetime.now()
    today_scans = [s for s in scans if s["timestamp"].date() == now.date()]

    # ✅ Daily totals for the last 7 days
    last_7_days = [(now - timedelta(days=i)).date() for i in range(6, -1, -1)]
    trend_data = []
    for d in last_7_days:
        count = sum(1 for s in scans if s["timestamp"].date() == d)
        avg_conf = round(
            sum(s["confidence"] for s in scans if s["timestamp"].date() == d) / count,
            2
        ) if count > 0 else 0
        trend_data.append({
            "date": d.strftime("%b %d"),
            "count": count,
            "avg_confidence": avg_conf
        })

    # ✅ Compute summary
    avg_confidence = round(sum(s["confidence"] for s in scans) / len(scans), 2)
    most_common = Counter(s["predicted_class"] for s in scans).most_common(1)[0][0]

    return jsonify({
        "total_today": len(today_scans),
        "avg_confidence": avg_confidence,
        "most_common": most_common,
        "trend": trend_data,
        "all_scans": scans
    })

# ============================================================
# 3️⃣ Prediction Endpoint (Confidence + Recommendation)
# ============================================================
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_path = "uploaded_image.png"
        file.save(image_path)

        from PIL import ImageStat
        image = Image.open(image_path).convert("L")
        img_array = np.array(image)

        brightness_mean = np.mean(img_array)
        brightness_std = np.std(img_array)
        stat_rgb = ImageStat.Stat(Image.open(image_path).convert("RGB"))
        mean_rgb = stat_rgb.mean
        diff_rgb = max(abs(mean_rgb[0]-mean_rgb[1]), abs(mean_rgb[1]-mean_rgb[2]), abs(mean_rgb[0]-mean_rgb[2]))

        if not (diff_rgb < 10 and 40 < brightness_mean < 200 and brightness_std > 30):
            return jsonify({
                "error": "This doesn't look like a chest X-ray image. Please upload a proper chest X-ray file."
            }), 400

        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return jsonify({"error": "Invalid X-ray format."}), 400

        results = predict(model, processed_img)
        if not results:
            return jsonify({"error": "No disease detected in the X-ray."}), 200

        # ✅ Extract top prediction
        top_label, top_conf = results[0]
        confidence = round(top_conf * 100, 2)

        # ✅ Build result dictionary BEFORE saving
        result = {
            "predicted_class": top_label,
            "confidence": confidence,
            "results": [
                {"disease": label, "confidence": round(prob * 100, 2)}
                for label, prob in results if prob > 0.6
            ]
        }

        # ✅ STEP: Save scan info for dashboard stats
        scans = load_scans()
        scans.append({
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().isoformat()
        })
        save_scans(scans)

        # ✅ Return the result to frontend
        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error during prediction."}), 500

# ============================================================
# 🧠 Grad-CAM generator function (define FIRST)
# ============================================================
def generate_gradcam(model, input_tensor):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import traceback
        from PIL import Image
        import torch.nn.functional as F
        from torchvision import models

        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device).requires_grad_(True)

        activations, gradients = [], []

        def forward_hook(module, inp, outp):
            activations.append(outp.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        # Hook the last major Swin stage
        target_layer = None
        for name, module in model.named_modules():
            if "layers.2" in name or "stage4" in name:
                target_layer = module
        if target_layer is None:
            print("⚠ No target layer found for Grad-CAM.")
            return None

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_backward_hook(backward_hook)

        outputs = model(input_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        probs = torch.sigmoid(outputs)[0]
        class_idx = torch.argmax(probs).item()

        model.zero_grad()
        probs[class_idx].backward(retain_graph=True)

        grad = gradients[-1].cpu().numpy()
        act = activations[-1].cpu().numpy()

        # -------------------------------
        # Swin-specific handling
        # -------------------------------
        if grad.ndim == 3:  # [B, N, C]
            grad = grad.mean(axis=1)  # [B, C]
            act = act.mean(axis=1)    # [B, C]
            weights = grad[0]

            # Ensure both are vectors
            if act.ndim == 1:
                act = act.reshape(1, -1)

            # Rebuild a 2D CAM map
            cam = np.dot(act, weights)
            cam = np.maximum(cam, 0).flatten()

            token_len = cam.shape[0]
            side = int(np.ceil(np.sqrt(token_len)))
            cam = np.pad(cam, (0, side * side - token_len), mode='constant')
            cam = cam.reshape(side, side)
        else:
            # Fallback for CNN-like layers
            weights = np.mean(grad, axis=(2, 3))[0]
            cam = np.maximum(np.sum(weights[:, None, None] * act[0], axis=0), 0)

        # Normalize & convert
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = (cam * 255).astype(np.uint8)

        handle_f.remove()
        handle_b.remove()

        print("✅ Grad-CAM generated successfully (robust Swin version).")
        return cam

    except Exception as e:
        print("❌ Grad-CAM error:", e)
        traceback.print_exc()
        return None
    # ============================================================
# 🧠 Guided Grad-CAM Utility (Sharper Visualization)
# ============================================================
class GuidedBackpropReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(positive_mask)
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        positive_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[positive_mask == 0] = 0
        return grad_input

def replace_relu_with_guided_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, GuidedBackpropReLU())
        else:
            replace_relu_with_guided_relu(module)

def combine_gradcam_guidedbackprop(gradcam, guided_backprop):
    guided_backprop = guided_backprop - guided_backprop.min()
    guided_backprop = guided_backprop / (guided_backprop.max() + 1e-8)
    heatmap = gradcam * guided_backprop
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return (heatmap * 255).astype(np.uint8)



# ============================================================
# 🌈 Final Enhanced Grad-CAM + Guided Backprop (Color-Corrected)
# ============================================================
@app.route('/gradcam', methods=['POST'])
def gradcam_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image_path = "gradcam_input.png"
    file.save(image_path)

    try:
        # Step 1 — Preprocess and run base Grad-CAM
        tensor = preprocess_image(image_path)
        cam = generate_gradcam(model, tensor)
        if cam is None:
            return jsonify({"error": "Grad-CAM failed"}), 500

        # Step 2 — Guided Backpropagation
        guided_model = model
        replace_relu_with_guided_relu(guided_model)
        tensor.requires_grad_(True)
        output = guided_model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        pred_class = torch.argmax(torch.sigmoid(output[0])).item()

        guided_model.zero_grad()
        output[0, pred_class].backward(retain_graph=True)

        guided_grad = tensor.grad.detach().cpu().numpy()[0]
        guided_grad = np.transpose(guided_grad, (1, 2, 0))
        guided_grad = np.mean(np.abs(guided_grad), axis=2)

        # Step 3 — Combine Grad-CAM and Guided Backprop
        combined_heatmap = combine_gradcam_guidedbackprop(cam, guided_grad)

        # Step 4 — Color normalization
        from scipy.ndimage import gaussian_filter
        low, high = np.percentile(combined_heatmap, (5, 99))
        combined_heatmap = np.clip((combined_heatmap - low) / (high - low), 0, 1)
        combined_heatmap = gaussian_filter(combined_heatmap, sigma=1.2)
        # ✅ Step 5 — Apply vivid colormap correctly (high contrast)
        import cv2
        cmap = plt.get_cmap("turbo")

        # Normalize Grad-CAM values between 0–1
        combined_heatmap = np.maximum(combined_heatmap, 0)
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() + 1e-8)
        combined_heatmap = (combined_heatmap * 255).astype(np.uint8)

        # Apply turbo/jet colormap
        heat_colored = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)
        heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)

        # Slightly boost saturation & contrast for visibility
        heat_colored = cv2.convertScaleAbs(heat_colored, alpha=1.3, beta=15)

        # Convert to PIL for blending
        heatmap_img = Image.fromarray(heat_colored).resize((224, 224))

        # Overlay on original image with enhanced color contrast
        orig = Image.open(image_path).convert("RGB").resize((224, 224))

        # 💡 Stronger color visibility
        overlay = Image.blend(orig, heatmap_img, alpha=0.25)  # 0.25 = more vivid heatmap (was 0.35)

        # 💡 Optional: boost red/yellow intensity and overall contrast
        from PIL import ImageEnhance
        overlay = ImageEnhance.Color(overlay).enhance(2.0)   # color saturation boost (was 1.6)
        overlay = ImageEnhance.Contrast(overlay).enhance(1.4)  # improves edge clarity

        # Optional: emphasize mid-range reds/yellows
        from PIL import ImageEnhance
        overlay = ImageEnhance.Contrast(overlay).enhance(1.4)
        overlay = ImageEnhance.Color(overlay).enhance(1.6)

        # Return image
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)
        print("✅ Grad-CAM vivid heatmap generated successfully.")
        return send_file(buf, mimetype="image/png")

        # Apply colormap via OpenCV to ensure proper color conversion
        import cv2
        heat_colored = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        heatmap_img = Image.fromarray(heat_colored).resize((224, 224))

        # Step 6 — Overlay heatmap on original image
        orig = Image.open(image_path).convert("RGB").resize((224, 224))
        overlay = Image.blend(orig, heatmap_img, alpha=0.5)

        # Step 7 — Optional: enhance contrast and brightness
        from PIL import ImageEnhance
        overlay = ImageEnhance.Contrast(overlay).enhance(1.4)
        overlay = ImageEnhance.Brightness(overlay).enhance(1.15)

        # Step 8 — Return as PNG
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)
        print("✅ Grad-CAM heatmap (OpenCV RGB version) generated successfully.")
        return send_file(buf, mimetype="image/png")
       
        print("✅ Grad-CAM heatmap (OpenCV RGB version) generated successfully.")
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        print(f"❌ Grad-CAM endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Grad-CAM failed"}), 500






# ============================================================
# 4️⃣ Nearby Pulmonologists (Google Maps API)
# ============================================================
GOOGLE_API_KEY = "AIzaSyCSy1oUBtfMefoerSKT8T0O26HHtDtlJ44"

@app.route('/nearby_pulmonologists', methods=['POST'])
def nearby_pulmonologists():
    try:
        data = request.get_json()
        lat = data.get("latitude")
        lng = data.get("longitude")

        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required"}), 400

        url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={lat},{lng}&radius=5000&type=doctor&keyword=pulmonologist&key={GOOGLE_API_KEY}"
        )

        response = requests.get(url)
        results = response.json().get("results", [])

        pulmonologists = []
        for place in results[:5]:
            pulmonologists.append({
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "rating": place.get("rating", "N/A"),
            })

        return jsonify({"pulmonologists": pulmonologists})

    except Exception as e:
        print("❌ Error fetching pulmonologists:", e)
        return jsonify({"error": str(e)}), 500

# ============================================================
# 5️⃣ Helper: Dynamic Report Filename
# ============================================================
def get_next_report_filename(base_dir="reports", base_name="VisioLung_Report"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [f for f in os.listdir(base_dir) if f.startswith(base_name) and f.lower().endswith(".pdf")]
    if not existing:
        filename = f"{base_name} (1).pdf"
    else:
        max_i = 0
        for f in existing:
            try:
                left, right = f.rfind("("), f.rfind(")")
                if left != -1 and right != -1 and right > left:
                    num = int(f[left+1:right])
                    max_i = max(max_i, num)
            except:
                continue
        filename = f"{base_name} ({max_i+1}).pdf"
    return os.path.join(base_dir, filename), filename

# ============================================================
# 6️⃣ Download PDF Report
# ============================================================
@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        from datetime import datetime
        from reportlab.platypus import Paragraph, Frame
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        data = request.get_json()
        name = data.get("name", "Unknown Patient")
        age = data.get("age", "N/A")
        gender = data.get("gender", "N/A")
        disease = data.get("disease", "Unknown")
        confidence = data.get("confidence", 0)
        timestamp = data.get("timestamp", datetime.now().strftime("%d/%m/%Y, %I:%M:%S %p"))

        safe_name = name.replace(" ", "_")
        report_path = f"VisioLung_Report_{safe_name}.pdf"

        c = canvas.Canvas(report_path, pagesize=A4)
        width, height = A4

        accent = (0.0, 0.45, 0.75)
        margin_left = 70
        y = height - 100

        logo_path = os.path.join("assets", "logo.png")
        if os.path.exists(logo_path):
            c.drawImage(logo_path, margin_left, y - 10, width=60, height=60, mask='auto')

        c.setFont("Times-Bold", 26)
        c.setFillColorRGB(*accent)
        c.drawString(margin_left + 80, y + 10, "VisioLung.AI")
        c.setFont("Times-Italic", 12)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawString(margin_left + 80, y - 10, "AI-Powered Chest X-Ray Diagnostic Report")

        c.setStrokeColorRGB(*accent)
        c.line(margin_left, y - 30, width - margin_left, y - 30)

        # Patient info
        y -= 70
        details = [("Patient Name:", name), ("Age:", age), ("Gender:", gender),
                   ("Predicted Disease:", disease), ("Confidence Level:", f"{confidence:.2f}%"),
                   ("Generated On:", timestamp)]
        for label, value in details:
            c.setFont("Times-Bold", 14)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(margin_left, y, label)
            c.setFont("Times-Roman", 14)
            c.setFillColorRGB(0.1, 0.3, 0.6)
            c.drawString(margin_left + 200, y, str(value))
            y -= 30

        # Note
        y -= 30
        styles = getSampleStyleSheet()
        note_style = ParagraphStyle(name='Note', fontName='Times-Roman', fontSize=12,
                                    leading=17, textColor=colors.black)
        note_text = (
            "<b><font color='#0077B6'>⚠ Important Note:</font></b><br/>"
            "This report is generated by an AI-based screening tool to assist professionals. "
            "Always consult a certified radiologist or pulmonologist for medical interpretation."
        )
        p = Paragraph(note_text, note_style)
        frame = Frame(margin_left, 100, width - 2 * margin_left, y - 100, showBoundary=0)
        frame.addFromList([p], c)

        c.setStrokeColorRGB(*accent)
        c.line(margin_left, 70, width - margin_left, 70)
        c.setFont("Times-Italic", 10)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawCentredString(width / 2, 55, "© 2025 VisioLung.AI — Research Use Only")
        c.save()

        return send_file(report_path, as_attachment=True, download_name=f"{report_path}")

    except Exception as e:
        print(f"❌ Report generation error: {e}")
        return jsonify({"error": "Failed to generate report"}), 500



# ============================================================
# 8️⃣ AI Chat Endpoint
# ============================================================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', '')
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Medical knowledge base for common questions
        medical_kb = {
            'infiltration': "Infiltration in a chest X-ray refers to areas where fluid, cells, or other substances have accumulated in the lung tissue, making it appear denser or whiter than normal. This can indicate conditions like pneumonia, pulmonary edema, or inflammation.",
            'pneumonia': "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms like cough, fever, and difficulty breathing. It's typically diagnosed through chest X-rays showing infiltrates or consolidation.",
            'effusion': "Pleural effusion is the accumulation of excess fluid in the space between the lungs and the chest wall (pleural space). On an X-ray, it appears as a white area at the bottom of the lung, often causing the lung to appear smaller.",
            'fibrosis': "Pulmonary fibrosis is a condition where lung tissue becomes damaged and scarred, making it thicker and stiffer. This makes it harder for the lungs to work properly. On X-rays, it appears as increased white markings or reticular patterns.",
            'atelectasis': "Atelectasis is the collapse or closure of a lung, resulting in reduced or absent gas exchange. It can appear as areas of increased density or reduced lung volume on chest X-rays.",
            'inflammation': "Lung inflammation can be reduced by: 1) Avoiding smoking and secondhand smoke, 2) Using air purifiers, 3) Staying hydrated, 4) Eating anti-inflammatory foods (fruits, vegetables, omega-3), 5) Getting regular exercise, 6) Avoiding pollutants and allergens, 7) Following prescribed medications from your doctor.",
        }
        
        # Simple keyword-based response (can be replaced with actual LLM API)
        message_lower = message.lower()
        response = None
        
        # General responses with priority
        if 'reduce' in message_lower and 'inflammation' in message_lower:
            response = medical_kb['inflammation']
        elif 'explain' in message_lower and 'diagnosis' in message_lower:
            if context:
                response = f"Based on your scan results, {context}This means the AI model has detected potential abnormalities in your chest X-ray. However, this is a screening tool and not a replacement for professional medical evaluation. Please consult with a qualified radiologist or pulmonologist for a comprehensive diagnosis and treatment plan."
            else:
                response = "To explain a diagnosis, please first upload and analyze an X-ray image. Then I can provide detailed explanations based on the results."
        elif 'what does' in message_lower:
            # Check if asking about a specific medical term
            found_term = False
            for term in medical_kb.keys():
                if term in message_lower:
                    response = medical_kb[term]
                    found_term = True
                    break
            if not found_term:
                response = "I can help explain medical terms related to chest X-rays and lung conditions. Try asking about specific terms like 'infiltration', 'pneumonia', 'effusion', or 'fibrosis'."
        else:
            # Check medical terms
            for term, explanation in medical_kb.items():
                if term in message_lower:
                    if context and ('explain' in message_lower or 'diagnosis' in message_lower):
                        response = f"Based on your current scan ({context}), {explanation}"
                    else:
                        response = explanation
                    break
            
            # If context exists and no specific term found, provide general diagnosis explanation
            if not response and context and ('explain' in message_lower or 'diagnosis' in message_lower or 'tell' in message_lower):
                response = f"Based on your scan results, {context}This indicates potential abnormalities detected by the AI. Please consult with a qualified healthcare professional for a comprehensive diagnosis and treatment plan."
            elif not response:
                # Default helpful response
                response = "I'm here to help answer questions about chest X-rays, lung conditions, and medical terminology. You can ask me to explain diagnoses, medical terms, or ask for general lung health advice. For specific medical concerns, please consult with a healthcare professional."
        
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Chat service unavailable"}), 500

# ============================================================
# 7️⃣ Root Route
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return "<h2>VisioLung.ai Backend is Running 🚀</h2>"

# ============================================================
# 🔟 Run Server
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
