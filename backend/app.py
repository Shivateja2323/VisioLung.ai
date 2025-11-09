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
        
        # All diseases supported by the model
        SUPPORTED_DISEASES = [
            "atelectasis", "cardiomegaly", "effusion", "infiltration",
            "mass", "nodule", "pneumonia", "pneumothorax",
            "consolidation", "edema", "emphysema", "fibrosis",
            "pleural thickening", "hernia"
        ]
        
        # Comprehensive medical knowledge base for all 14 diseases
        medical_kb = {
            'atelectasis': {
                'description': "Atelectasis is the collapse or closure of a lung, resulting in reduced or absent gas exchange. It occurs when the tiny air sacs (alveoli) in the lung deflate or become filled with fluid.",
                'xray_appearance': "On chest X-rays, atelectasis appears as areas of increased density, reduced lung volume, or displacement of structures like the heart or diaphragm.",
                'causes': "Common causes include blockage of airways, pressure on the lung from outside, fluid accumulation, or post-surgical complications.",
                'symptoms': "Symptoms may include difficulty breathing, chest pain, and coughing. However, small areas of atelectasis may be asymptomatic.",
                'treatment': "Treatment depends on the cause and may include deep breathing exercises, chest physiotherapy, removal of airway blockages, or addressing underlying conditions."
            },
            'cardiomegaly': {
                'description': "Cardiomegaly is an enlargement of the heart. It's not a disease itself but rather a sign of an underlying condition affecting the heart.",
                'xray_appearance': "On chest X-rays, cardiomegaly appears as an enlarged heart shadow. The cardiothoracic ratio (heart width to chest width) is typically greater than 50%.",
                'causes': "Common causes include high blood pressure, heart valve disease, cardiomyopathy, coronary artery disease, and fluid around the heart (pericardial effusion).",
                'symptoms': "Symptoms may include shortness of breath, fatigue, chest pain, irregular heartbeat, and swelling in the legs or abdomen.",
                'treatment': "Treatment focuses on managing the underlying cause and may include medications (ACE inhibitors, beta-blockers), lifestyle changes, or in severe cases, surgery."
            },
            'effusion': {
                'description': "Pleural effusion is the accumulation of excess fluid in the space between the lungs and the chest wall (pleural space). This space normally contains a small amount of fluid for lubrication.",
                'xray_appearance': "On chest X-rays, pleural effusion appears as a white area at the bottom of the lung, often causing the lung to appear smaller and creating a meniscus (curved) appearance.",
                'causes': "Causes include congestive heart failure, pneumonia, cancer, pulmonary embolism, kidney disease, and autoimmune conditions.",
                'symptoms': "Symptoms include shortness of breath, chest pain (especially when breathing), dry cough, and difficulty lying flat.",
                'treatment': "Treatment depends on the cause and may include draining the fluid (thoracentesis), treating the underlying condition, or in some cases, surgery."
            },
            'infiltration': {
                'description': "Infiltration in a chest X-ray refers to areas where fluid, cells, or other substances have accumulated in the lung tissue, making it appear denser or whiter than normal.",
                'xray_appearance': "On chest X-rays, infiltrates appear as patchy or diffuse white areas in the lung fields, indicating abnormal density.",
                'causes': "Common causes include pneumonia, pulmonary edema (fluid in lungs), inflammation, infection, or bleeding into the lung tissue.",
                'symptoms': "Symptoms vary by cause but may include cough, fever, shortness of breath, chest pain, and fatigue.",
                'treatment': "Treatment depends on the underlying cause - antibiotics for infections, diuretics for fluid overload, or addressing the specific condition causing the infiltration."
            },
            'mass': {
                'description': "A lung mass is an abnormal growth or lesion in the lung tissue that appears as a distinct, well-defined area on imaging.",
                'xray_appearance': "On chest X-rays, masses appear as well-defined, rounded or irregular opacities that are typically larger than 3 cm in diameter.",
                'causes': "Causes can be benign (non-cancerous) such as granulomas, hamartomas, or infections, or malignant (cancerous) such as lung cancer.",
                'symptoms': "Symptoms may include persistent cough, chest pain, coughing up blood, weight loss, and shortness of breath, though some masses are asymptomatic.",
                'treatment': "Treatment depends on whether the mass is benign or malignant. Benign masses may be monitored, while malignant masses require cancer treatment including surgery, chemotherapy, or radiation."
            },
            'nodule': {
                'description': "A lung nodule is a small, round or oval-shaped growth in the lung tissue, typically smaller than 3 cm in diameter.",
                'xray_appearance': "On chest X-rays, nodules appear as small, well-defined, round or oval opacities in the lung fields.",
                'causes': "Most nodules are benign and can be caused by old infections (granulomas), scar tissue, or inflammation. Some may be early-stage cancers.",
                'symptoms': "Most small nodules are asymptomatic and are discovered incidentally on imaging. Larger nodules may cause symptoms similar to masses.",
                'treatment': "Small nodules are often monitored with follow-up imaging. Larger or suspicious nodules may require biopsy or removal, depending on size, growth, and appearance."
            },
            'pneumonia': {
                'description': "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing difficulty breathing.",
                'xray_appearance': "On chest X-rays, pneumonia appears as areas of consolidation (white, dense areas) or infiltrates in the lung tissue, often in a specific pattern depending on the type of pneumonia.",
                'causes': "Caused by bacteria, viruses, or fungi. Common bacterial causes include Streptococcus pneumoniae, while viral causes include influenza and COVID-19.",
                'symptoms': "Symptoms include cough (with or without phlegm), fever, chills, difficulty breathing, chest pain, fatigue, and sometimes nausea or vomiting.",
                'treatment': "Treatment depends on the cause: antibiotics for bacterial pneumonia, antiviral medications for viral pneumonia, rest, fluids, and supportive care. Severe cases may require hospitalization."
            },
            'pneumothorax': {
                'description': "Pneumothorax is the presence of air in the pleural space (between the lung and chest wall), causing the lung to collapse partially or completely.",
                'xray_appearance': "On chest X-rays, pneumothorax appears as a dark area (air) with no lung markings, and the edge of the collapsed lung may be visible. The lung appears smaller.",
                'causes': "Can be spontaneous (occurring without injury, often in tall, thin young men) or traumatic (from injury, medical procedures, or underlying lung disease).",
                'symptoms': "Symptoms include sudden sharp chest pain, shortness of breath, rapid heart rate, and in severe cases, cyanosis (bluish skin) and shock.",
                'treatment': "Small pneumothoraces may resolve on their own. Larger ones require removal of air via needle aspiration or chest tube insertion. Recurrent cases may need surgery."
            },
            'consolidation': {
                'description': "Consolidation occurs when the air spaces in the lungs are filled with fluid, pus, blood, or cells, replacing the normal air-filled spaces.",
                'xray_appearance': "On chest X-rays, consolidation appears as dense, white areas that obscure normal lung markings and blood vessels.",
                'causes': "Common causes include pneumonia, pulmonary edema (heart failure), lung cancer, and bleeding into the lungs.",
                'symptoms': "Symptoms include cough, fever, difficulty breathing, chest pain, and production of sputum, depending on the underlying cause.",
                'treatment': "Treatment targets the underlying cause: antibiotics for infection, diuretics for fluid overload, or specific treatments for cancer or other conditions."
            },
            'edema': {
                'description': "Pulmonary edema is the accumulation of fluid in the air spaces and tissues of the lungs, making breathing difficult.",
                'xray_appearance': "On chest X-rays, pulmonary edema appears as fluffy, bilateral infiltrates, often in a 'bat wing' pattern, with enlarged heart and fluid in the pleural spaces.",
                'causes': "Most commonly caused by heart failure (cardiogenic), but can also result from kidney failure, severe infections, drug reactions, or high altitudes.",
                'symptoms': "Symptoms include severe shortness of breath, difficulty breathing when lying flat, anxiety, coughing up pink, frothy sputum, and chest pain.",
                'treatment': "Treatment includes oxygen therapy, diuretics to remove excess fluid, medications to improve heart function, and addressing the underlying cause. This is a medical emergency."
            },
            'emphysema': {
                'description': "Emphysema is a chronic lung condition where the air sacs (alveoli) are damaged and enlarged, reducing the surface area for gas exchange.",
                'xray_appearance': "On chest X-rays, emphysema appears as hyperinflated lungs (lungs appear larger), flattened diaphragm, and decreased lung markings. The chest may appear barrel-shaped.",
                'causes': "Most commonly caused by long-term smoking. Other causes include exposure to air pollution, chemical fumes, dust, and rare genetic conditions (alpha-1 antitrypsin deficiency).",
                'symptoms': "Symptoms include shortness of breath (especially with exertion), chronic cough, wheezing, chest tightness, and fatigue. Symptoms develop gradually over years.",
                'treatment': "Treatment includes smoking cessation, bronchodilators, inhaled steroids, oxygen therapy, pulmonary rehabilitation, and in severe cases, lung volume reduction surgery or lung transplant."
            },
            'fibrosis': {
                'description': "Pulmonary fibrosis is a condition where lung tissue becomes damaged and scarred, making it thicker and stiffer. This makes it harder for the lungs to work properly.",
                'xray_appearance': "On chest X-rays, fibrosis appears as increased white markings (reticular patterns), honeycombing in advanced cases, and reduced lung volume.",
                'causes': "Causes include long-term exposure to toxins (asbestos, silica), radiation therapy, certain medications, autoimmune diseases, and idiopathic (unknown cause) pulmonary fibrosis.",
                'symptoms': "Symptoms include progressive shortness of breath, dry cough, fatigue, unexplained weight loss, and clubbing of fingers and toes.",
                'treatment': "Treatment includes medications to slow progression (pirfenidone, nintedanib), oxygen therapy, pulmonary rehabilitation, and in some cases, lung transplant. There is no cure, but treatment can slow progression."
            },
            'pleural thickening': {
                'description': "Pleural thickening is the scarring and thickening of the pleura (the membrane covering the lungs and lining the chest cavity), often due to inflammation or injury.",
                'xray_appearance': "On chest X-rays, pleural thickening appears as irregular, thickened white lines along the edges of the lungs or chest wall.",
                'causes': "Common causes include asbestos exposure, previous infections (tuberculosis, empyema), trauma, radiation therapy, and inflammatory conditions.",
                'symptoms': "Many cases are asymptomatic. When symptoms occur, they may include shortness of breath, chest pain, and reduced lung function.",
                'treatment': "Treatment depends on the cause and severity. Mild cases may be monitored, while severe cases affecting lung function may require surgical intervention (pleurectomy)."
            },
            'hernia': {
                'description': "A diaphragmatic hernia occurs when abdominal organs protrude through an opening in the diaphragm into the chest cavity.",
                'xray_appearance': "On chest X-rays, a hernia appears as abnormal shadows or opacities in the lower chest, often with displacement of normal structures.",
                'causes': "Can be congenital (present at birth) or acquired through trauma, surgery, or increased pressure in the abdomen.",
                'symptoms': "Symptoms vary but may include difficulty breathing, chest pain, heartburn, difficulty swallowing, and in severe cases, respiratory distress.",
                'treatment': "Treatment depends on severity. Small, asymptomatic hernias may be monitored, while larger or symptomatic hernias typically require surgical repair."
            }
        }
        
        # Keywords that indicate out-of-scope questions (non-lung/chest X-ray related)
        out_of_scope_keywords = [
            'diabetes', 'diabetic', 'blood sugar', 'insulin', 'glucose',
            'heart attack', 'stroke', 'brain', 'neurological', 'headache', 'migraine',
            'kidney', 'liver', 'stomach', 'digestive', 'gastrointestinal', 'ulcer',
            'bone', 'fracture', 'arthritis', 'joint', 'spine', 'back pain',
            'eye', 'vision', 'retina', 'cataract',
            'skin', 'rash', 'dermatology', 'acne',
            'cancer'  # Too general - we'll check if it's lung cancer specifically
        ]
        
        # Lung/chest X-ray related keywords (in scope)
        in_scope_keywords = [
            'lung', 'chest', 'x-ray', 'xray', 'pulmonary', 'respiratory', 'breathing',
            'atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass', 'nodule',
            'pneumonia', 'pneumothorax', 'consolidation', 'edema', 'emphysema',
            'fibrosis', 'pleural', 'thickening', 'hernia', 'alveoli', 'bronchi',
            'diagnosis', 'scan', 'image', 'radiology', 'radiologist'
        ]
        
        message_lower = message.lower()
        response = None
        
        # Check if question is out of scope
        is_out_of_scope = False
        out_of_scope_term = None
        
        # Check for out-of-scope terms (but allow if lung/chest context is present)
        for term in out_of_scope_keywords:
            if term in message_lower:
                # Check if there's lung/chest context
                has_lung_context = any(keyword in message_lower for keyword in in_scope_keywords)
                if not has_lung_context:
                    is_out_of_scope = True
                    out_of_scope_term = term
                    break
        
        # If out of scope, return error message
        if is_out_of_scope:
            response = f"❌ I'm sorry, but I can only answer questions related to chest X-rays and lung conditions detected by this system. Your question seems to be about {out_of_scope_term}, which is outside my scope.\n\nPlease ask relevant questions about:\n• Chest X-ray interpretation\n• Lung diseases (Atelectasis, Pneumonia, Effusion, Fibrosis, etc.)\n• Medical terms related to chest imaging\n• Your X-ray scan results\n\nI can help explain diagnoses, medical terminology, and provide information about the 14 lung conditions this system can detect."
            return jsonify({"response": response})
        
        # Check if asking about a specific disease
        found_disease = None
        for disease in SUPPORTED_DISEASES:
            if disease in message_lower:
                found_disease = disease
                break
        
        # Handle different question types
        if found_disease:
            disease_info = medical_kb[found_disease]
            
            # Determine what aspect they're asking about
            if any(word in message_lower for word in ['what is', 'what does', 'explain', 'tell me about', 'meaning']):
                response = f"**{found_disease.capitalize()}**\n\n{disease_info['description']}\n\n**X-ray Appearance:** {disease_info['xray_appearance']}\n\n**Common Causes:** {disease_info['causes']}\n\n**Symptoms:** {disease_info['symptoms']}\n\n**Treatment:** {disease_info['treatment']}"
                
                if context:
                    response = f"Based on your scan results showing {context.strip()}, here's information about **{found_disease.capitalize()}**:\n\n{response}\n\n⚠️ **Important:** This is AI-assisted screening. Please consult with a qualified radiologist or pulmonologist for professional medical interpretation and treatment recommendations."
            elif any(word in message_lower for word in ['symptom', 'sign', 'feel', 'experience']):
                response = f"**Symptoms of {found_disease.capitalize()}:**\n\n{disease_info['symptoms']}"
            elif any(word in message_lower for word in ['cause', 'why', 'reason', 'due to']):
                response = f"**Causes of {found_disease.capitalize()}:**\n\n{disease_info['causes']}"
            elif any(word in message_lower for word in ['treatment', 'treat', 'cure', 'medicine', 'medication', 'therapy']):
                response = f"**Treatment for {found_disease.capitalize()}:**\n\n{disease_info['treatment']}\n\n⚠️ **Important:** Always consult with a healthcare professional for proper diagnosis and treatment. This information is for educational purposes only."
            elif any(word in message_lower for word in ['x-ray', 'xray', 'appearance', 'look like', 'show', 'detect']):
                response = f"**X-ray Appearance of {found_disease.capitalize()}:**\n\n{disease_info['xray_appearance']}"
            else:
                # General information about the disease
                response = f"**{found_disease.capitalize()}**\n\n{disease_info['description']}\n\n**X-ray Appearance:** {disease_info['xray_appearance']}"
                
        # Handle diagnosis explanation requests
        elif any(word in message_lower for word in ['explain', 'diagnosis', 'result', 'scan', 'what does this mean']):
            if context:
                response = f"Based on your scan results: {context.strip()}\n\nThis indicates that the AI model has detected potential abnormalities in your chest X-ray. The system analyzes X-ray images to identify signs of various lung conditions.\n\n⚠️ **Important:** This is an AI-assisted screening tool and not a replacement for professional medical evaluation. Please consult with a qualified radiologist or pulmonologist for:\n• Comprehensive diagnosis\n• Detailed interpretation of your X-ray\n• Treatment recommendations\n• Follow-up care\n\nWould you like me to explain more about the detected condition?"
            else:
                response = "To explain a diagnosis, please first upload and analyze an X-ray image. Once you have scan results, I can provide detailed explanations about the detected conditions.\n\nYou can ask me questions like:\n• 'Explain this diagnosis'\n• 'What does [disease name] mean?'\n• 'Tell me about pneumonia'\n• 'What are the symptoms of effusion?'"
        
        # Handle general lung health questions
        elif any(word in message_lower for word in ['lung health', 'healthy lungs', 'lung care', 'prevent', 'reduce inflammation']):
            if 'inflammation' in message_lower:
                response = "**Ways to Reduce Lung Inflammation:**\n\n1. **Avoid smoking and secondhand smoke** - This is the most important step\n2. **Use air purifiers** - Especially in areas with poor air quality\n3. **Stay hydrated** - Drink plenty of water\n4. **Eat anti-inflammatory foods** - Fruits, vegetables, omega-3 rich foods\n5. **Get regular exercise** - Improves lung function\n6. **Avoid pollutants and allergens** - Use masks in polluted areas\n7. **Follow prescribed medications** - As directed by your doctor\n8. **Practice deep breathing exercises** - Can help improve lung capacity\n\n⚠️ If you're experiencing persistent lung inflammation, please consult with a healthcare professional."
            else:
                response = "**Tips for Maintaining Lung Health:**\n\n1. **Don't smoke** - Avoid all tobacco products\n2. **Exercise regularly** - Improves lung capacity and function\n3. **Avoid air pollution** - Use air purifiers, avoid high-pollution areas\n4. **Practice deep breathing** - Strengthens respiratory muscles\n5. **Get regular check-ups** - Especially if you have risk factors\n6. **Stay hydrated** - Helps keep lung tissue healthy\n7. **Eat a healthy diet** - Rich in antioxidants and anti-inflammatory foods\n8. **Get vaccinated** - Flu and pneumonia vaccines can protect lung health\n\n⚠️ For personalized advice, please consult with a healthcare professional."
        
        # Handle questions about what the system can do
        elif any(word in message_lower for word in ['what can you', 'help', 'capabilities', 'what do you', 'assist']):
            response = "I'm your AI medical assistant specialized in chest X-ray analysis. I can help you with:\n\n✅ **Explain diagnoses** - Understand what your X-ray results mean\n✅ **Medical terminology** - Learn about lung conditions and X-ray findings\n✅ **Disease information** - Get details about the 14 conditions I can detect:\n   • Atelectasis, Cardiomegaly, Effusion, Infiltration\n   • Mass, Nodule, Pneumonia, Pneumothorax\n   • Consolidation, Edema, Emphysema, Fibrosis\n   • Pleural Thickening, Hernia\n✅ **X-ray interpretation** - Understand how conditions appear on X-rays\n✅ **General lung health** - Tips for maintaining healthy lungs\n\n⚠️ **Important:** I provide educational information only. Always consult with qualified healthcare professionals for medical diagnosis and treatment."
        
        # Handle general greetings
        elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "👋 Hello! I'm your AI medical assistant specialized in chest X-ray analysis. I can help explain diagnoses, medical terms, and answer questions about lung health and the 14 conditions this system can detect.\n\nHow can I assist you today? You can ask me:\n• About your X-ray results\n• To explain medical terms\n• Questions about specific lung conditions\n• General lung health tips"
        
        # If no specific match found, check if it's a general medical question
        elif any(word in message_lower for word in ['medical', 'doctor', 'hospital', 'treatment', 'medicine']):
            # Check if it's related to lungs/chest
            if any(keyword in message_lower for keyword in in_scope_keywords):
                response = "I can help answer questions about chest X-rays and lung conditions. Could you please be more specific? For example:\n• 'What is pneumonia?'\n• 'Explain my diagnosis'\n• 'What does effusion mean?'\n• 'Tell me about lung nodules'"
            else:
                response = "❌ I'm specialized in chest X-ray analysis and lung conditions only. Please ask relevant questions about:\n• Chest X-ray interpretation\n• Lung diseases (the 14 conditions this system detects)\n• Medical terms related to chest imaging\n• Your X-ray scan results\n\nFor questions about other medical topics, please consult with a healthcare professional."
        
        # Default response for unrecognized but potentially relevant questions
        else:
            # Check if it contains any lung/chest related keywords
            if any(keyword in message_lower for keyword in in_scope_keywords):
                response = "I understand you're asking about chest X-rays or lung health. Could you please rephrase your question? For example:\n• 'What is [disease name]?'\n• 'Explain my diagnosis'\n• 'Tell me about pneumonia'\n• 'What are the symptoms of effusion?'\n\nI can provide information about the 14 lung conditions this system can detect and help interpret X-ray results."
            else:
                response = "❌ I'm sorry, but I can only answer questions related to chest X-rays and lung conditions. Your question seems to be outside my scope.\n\nPlease ask relevant questions about:\n• Chest X-ray interpretation and analysis\n• Lung diseases (Atelectasis, Pneumonia, Effusion, Fibrosis, Mass, Nodule, etc.)\n• Medical terms related to chest imaging\n• Your X-ray scan results\n• General lung health\n\nI can help explain diagnoses, medical terminology, and provide information about the 14 lung conditions this system can detect."
        
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
