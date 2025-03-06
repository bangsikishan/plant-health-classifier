import io
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)

CORS(app)


MODEL_SAVE_PATH = "D:\\Machine Learning\\image_classification\\image_classification_model\\distilled_model_1.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 model
student = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
student.classifier[1] = nn.Linear(student.classifier[1].in_features, 3)

# Load model weights
student.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
student.eval()
student.to(device)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = ["angular_leaf_spot", "bean_rust", "healthy"]


@app.route("/")
def index():
    return send_from_directory(os.getcwd(), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict the class of an uploaded image."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Read image file
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = student(input_batch)

        _, predicted = torch.max(output.data, 1)
        predicted_class = class_names[predicted.item()]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
