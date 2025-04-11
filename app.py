import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file,jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import csv
import re
import cv2
from skimage.segmentation import mark_boundaries
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from lime import lime_image
import time
import threading
from matplotlib.colors import LinearSegmentedColormap

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Define the model class
class DenseNetClassifier(nn.Module):
    def __init__(self):
        super(DenseNetClassifier, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        num_features = self.densenet.classifier.in_features

        # Adding AdaptiveAvgPool to match input size for linear layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classify diseases (multi-label) and view position (single-label)
        self.densenet.classifier = nn.Linear(num_features, 20)  # For 20 diseases
        self.view_classifier = nn.Linear(num_features, 2)  # PA or AP (binary classification)

    def forward(self, x):
        # Pass through densenet features
        features = self.densenet.features(x)
        features = torch.relu(features)

        # Apply adaptive pooling and flatten
        pooled_features = self.avg_pool(features)
        features = torch.flatten(pooled_features, 1)

        # Disease and view classification outputs
        disease_out = self.densenet.classifier(features)
        view_out = self.view_classifier(features)
        return disease_out, view_out


# Define the model architecture (same as used during training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNetClassifier().to(device)
model.load_state_dict(torch.load('densenet_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the transformation pipeline for the image (match your training transformation)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to predict the class of an image
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        disease_out, view_out = model(img_tensor)

    _, predicted_disease = torch.max(disease_out, 1)
    _, predicted_view = torch.max(view_out, 1)

    disease_classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                       'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                       'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax', 'Pneumoperitoneum',
                       'Pneumomediastinum', 'Subcutaneous Emphysema', 'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding']

    view_classes = ['PA', 'AP']

    predicted_disease_label = disease_classes[predicted_disease.item()]
    predicted_view_label = view_classes[predicted_view.item()]

    return f"Disease: {predicted_disease_label}, View: {predicted_view_label}"


# Function to generate LIME explanation for the uploaded image
def generate_lime_explanation(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    def predict_fn(image):
        image_tensor = torch.tensor(image).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.no_grad():
            disease_out, _ = model(image_tensor)
        return torch.nn.functional.softmax(disease_out, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=30000  # Reduced for faster computation
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=20,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp, mask)
    plt.imshow(lime_img)
    plt.axis('off')
    lime_image_path = 'static/uploads/lime_explanation.png'
    plt.savefig(lime_image_path, bbox_inches='tight')
    plt.close()

    return lime_image_path


# Grad-CAM Implementation (ensure it's on GPU)
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.to(device)
        self.target_layer = target_layer or model.densenet.features[-1]  # Use the last conv layer as the default

        # Placeholder for gradients and activations
        self.gradients = None
        self.activations = None

        # Register hooks for gradients and activations
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Save the output of the forward pass (activations)
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            # Save the gradients
            self.gradients = grad_out[0]

        # Register forward and backward hooks on the target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        # Get the score for the class we want (e.g., max predicted class)
        class_idx = torch.argmax(output[0]).item()

        # Backward pass to get gradients
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Get the activations and gradients
        activations = self.activations[0]
        gradients = self.gradients[0]

        # Pool the gradients across all spatial dimensions (global average pooling)
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        # Apply ReLU to the result
        cam = torch.clamp(cam, min=0)

        # Normalize the result
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        # Convert to numpy
        return cam.cpu().detach().numpy()


# Function to generate Grad-CAM visualization
def generate_gradcam(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Grad-CAM
    gradcam = GradCAM(model)
    gradcam_map = gradcam.generate_cam(img_tensor)

    # Resize the Grad-CAM output to the image size
    gradcam_map = cv2.resize(gradcam_map, (img.size[0], img.size[1]))
    gradcam_map = np.uint8(255 * gradcam_map)

    # Apply heatmap to original image
    heatmap = cv2.applyColorMap(gradcam_map, cv2.COLORMAP_JET)
    original_img = np.array(img)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return superimposed_img


# PDF report generation
def generate_pdf_report(image_path, prediction, lime_explanation, gradcam_path, patient_details):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 18, txt="Chest X-ray Report", ln=True, align='C')
    pdf.ln(10)

    # Patient Details Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 15, txt="Patient Details:", ln=True)

    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt=f"Name: {patient_details[0]}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {patient_details[1]}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {patient_details[2]}", ln=True)
    pdf.cell(200, 10, txt=f"Contact Number: {patient_details[6]}", ln=True)
    pdf.cell(200, 10, txt=f"Lab Name: {patient_details[7]}", ln=True)
    pdf.cell(200, 10, txt=f"Exam Date: {patient_details[3]}", ln=True)
    pdf.cell(200, 10, txt=f"Exam Time: {patient_details[4]}", ln=True)
    pdf.cell(200, 10, txt=f"Exam Type: {patient_details[8]}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Medical History: {patient_details[5]}", ln=True)
    pdf.ln(10)

    # Add patient details here (could be added dynamically)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 15, txt=f"Prediction:\n {prediction}", ln=True)
    pdf.ln(10)

    # Add Grad-CAM Image
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Grad-CAM Visualization:", ln=True)
    pdf.image(gradcam_path, w=150)

    # Add LIME Explanation
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="LIME Explanation:", ln=True)
    pdf.image(lime_explanation, w=150)

    # Output PDF
    pdf_output_path = 'static/uploads/report.pdf'
    pdf.output(pdf_output_path)
    return pdf_output_path


# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the image upload page
@app.route('/')
def upload_form():
    return render_template('upload_image.html')

# Route to handle file upload and redirect to patient details input
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if 'file' not in request.files or file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Redirect to patient details input form
        return redirect(url_for('patient_details', image_path=file_path))

    flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.')
    return redirect(url_for('upload_form'))



# Route for patient details input form
@app.route('/patient_details')
def patient_details():
    image_path = request.args.get('image_path')
    return render_template('patient_details.html', image_path=image_path)


# Route to save patient details and analyze image
@app.route('/save_patient_details', methods=['POST'])
def save_patient_details():
    # Collect patient details from form
    patient_name = request.form['name']
    patient_age = request.form['age']
    patient_gender = request.form['gender']
    exam_date = request.form['examDate']
    exam_time = request.form['examTime']
    medical_history = request.form['history']
    contact_number = request.form['contact']
    lab_name = request.form['labName']
    exam_type = request.form['examType']
    image_path = request.form['image_path']

    # Bundle patient details
    patient_details=[patient_name, patient_age, patient_gender, exam_date, exam_time, medical_history, contact_number, lab_name, exam_type]

    # Redirect to loading page while processing
    return render_template("loading.html", image_path=image_path, patient_details=patient_details)


@app.route('/generate_result')
def generate_result():
    # Get image path and patient details from URL parameters
    image_path = request.args.get('image_path')
    patient_details_json = request.args.get('patient_details')

    # Deserialize patient details from JSON to a list
    try:
        patient_details = json.loads(patient_details_json)  # Deserialize JSON
        if not isinstance(patient_details, list):
            return "Error: Patient details must be a list", 400
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for patient details", 400

    # Validate length of patient_details
    if len(patient_details) < 8:
        return "Error: Insufficient patient details", 400

    # Model analysis
    prediction = predict_image(image_path)
    lime_explanation = generate_lime_explanation(image_path)
    gradcam_image = generate_gradcam(image_path)
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_output.png')
    cv2.imwrite(gradcam_path, gradcam_image)

    # Generate PDF report
    pdf_path = generate_pdf_report(
        image_path, prediction, lime_explanation, gradcam_path, patient_details)

    # Render result page
    return render_template('result.html', pdf_path=pdf_path, prediction=prediction)


# Route to handle PDF download
@app.route('/download_pdf')
def download_pdf():
    pdf_path = request.args.get('pdf_path')
    if pdf_path and os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')
    else:
        flash('File not found or path is invalid.')
        return redirect(url_for('upload_form'))


# if __name__ == '__main__':
#     app.run(debug=True, port=7007)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5005)))
