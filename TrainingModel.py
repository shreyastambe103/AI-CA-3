import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
import cv2
from lime import lime_image
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Preprocessing and Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random flip for augmentation
    transforms.RandomRotation(1),  # Random rotation for augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # To make the column numeric type
        self.data_frame['view_position_encoded'] = self.data_frame['View Position'].map({'AP': 0, 'PA': 1})

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        disease_labels = torch.tensor(self.data_frame.iloc[idx, 1:21].astype(np.float32).values,
                                       dtype=torch.float32)

        # Fetch view label as a tensor
        view_label = torch.tensor(self.data_frame.iloc[idx]['view_position_encoded'], dtype=torch.long)

        return image, disease_labels, view_label

# Function to collate data and skip None entries
def collate_fn(batch):
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    return list(zip(*batch)) if batch else ([], [], [])  # Return empty lists if no valid items

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

# Step 2: Define DataLoader for train, val
train_dataset = ChestXrayDataset(csv_file=r"C:\Users\shreya\PycharmProjects\DL Project\train_final.csv", root_dir=r"C:\Users\shreya\PycharmProjects\DL Project\extracted_images\train", transform=transform)
val_dataset = ChestXrayDataset(csv_file=r"C:\Users\shreya\PycharmProjects\DL Project\val_final.csv", root_dir=r"C:\Users\shreya\PycharmProjects\DL Project\extracted_images\val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Step 3: Define DenseNet Model for Disease Classification & View Position
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 4: Initialize the model, loss functions, optimizer
class_weights_tensor = torch.load(r"C:\Users\shreya\PycharmProjects\DL Project\class_weights_tensor.pth")
model = DenseNetClassifier().to(device)
disease_criterion = nn.BCEWithLogitsLoss(weight=class_weights_tensor.to(device))  # For multi-label disease classification
view_criterion = nn.CrossEntropyLoss()  # For view classification

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, disease_labels, view_label in train_loader:
        images=images.to(device)
        disease_labels = disease_labels.to(device)  # Move labels to GPU
        view_label = view_label.to(device)

        optimizer.zero_grad()
        disease_pred, view_pred = model(images)

        disease_loss = disease_criterion(disease_pred, disease_labels)
        view_loss = view_criterion(view_pred, view_label)
        loss = disease_loss + view_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')


# Step 7: Validation and Accuracy Calculation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, disease_labels, view_label in val_loader:
        images = images.to(device)  # Move images to GPU
        disease_labels = disease_labels.to(device)

        disease_pred, view_pred = model(images)
        disease_pred_labels = (torch.sigmoid(disease_pred) > 0.2).int()
        all_preds.extend(disease_pred_labels.cpu().numpy())
        all_labels.extend(disease_labels.cpu().numpy())

# Calculate validation accuracy
accuracy = accuracy_score(np.array(all_labels).flatten(), np.array(all_preds).flatten())
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Step 8: Save the trained model
torch.save(model.state_dict(), 'densenet_model.pth')
print("Model saved successfully.")




# Test Accuracy
# Load the test dataset
test_dataset = ChestXrayDataset(csv_file=r'C:\Users\shreya\PycharmProjects\DL Project\test_final.csv', root_dir=r'C:\Users\shreya\PycharmProjects\DL Project\extracted_images\test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained model
model = DenseNetClassifier().to(device)  # Recreate the model architecture
model.load_state_dict(torch.load('densenet_model.pth'))# Load the saved weights
model.to(device)
model.eval()

# Initialize lists to store true labels and predictions
true_disease_labels = []
pred_disease_labels = []
true_view_labels = []
pred_view_labels = []

# Step 1: Perform inference on the test set
with torch.no_grad():
    for images, disease_labels, view_label in test_loader:
        images = images.to(device)
        disease_labels = disease_labels.to(device)  # Move labels to GPU
        view_label = view_label.to(device)

        disease_pred, view_pred = model(images)

        # Convert the sigmoid outputs into binary predictions (for disease classification)
        disease_pred_labels = (torch.sigmoid(disease_pred) > 0.2).int()

        # Store predictions and true labels for both disease and view
        pred_disease_labels.extend(disease_pred_labels.cpu().numpy())
        true_disease_labels.extend(disease_labels.cpu().numpy())

        # For view classification, get the argmax of the output
        pred_view_labels.extend(torch.argmax(view_pred, dim=1).cpu().numpy())
        true_view_labels.extend(view_label.cpu().numpy())

# Step 2: Calculate Accuracy, Precision, Recall, F1 for Disease Classification
accuracy = accuracy_score(np.array(true_disease_labels).flatten(), np.array(pred_disease_labels).flatten())
precision = precision_score(np.array(true_disease_labels).flatten(), np.array(pred_disease_labels).flatten(), average='macro')
recall = recall_score(np.array(true_disease_labels).flatten(), np.array(pred_disease_labels).flatten(), average='macro')
f1 = f1_score(np.array(true_disease_labels).flatten(), np.array(pred_disease_labels).flatten(), average='macro')
roc_auc = roc_auc_score(np.array(true_disease_labels), np.array(pred_disease_labels), average='macro')

print(f"Disease Classification Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")

# Step 3: Calculate Accuracy for View Classification (PA/AP)
view_accuracy = accuracy_score(true_view_labels, pred_view_labels)
view_precision = precision_score(true_view_labels, pred_view_labels, average='macro')
view_recall = recall_score(true_view_labels, pred_view_labels, average='macro')
view_f1 = f1_score(true_view_labels, pred_view_labels, average='macro')

print(f"\nView Classification (PA/AP) Metrics:")
print(f"Accuracy: {view_accuracy * 100:.2f}%")
print(f"Precision: {view_precision:.2f}")
print(f"Recall: {view_recall:.2f}")
print(f"F1-score: {view_f1:.2f}")

df = pd.read_csv(r'C:\Users\shreya\PycharmProjects\DL Project\train_final.csv')
disease_columns = list(df.columns[1:21])
print(classification_report(np.array(true_disease_labels), np.array(pred_disease_labels), target_names=disease_columns))


# cm = confusion_matrix(np.array(true_disease_labels).flatten(), np.array(pred_disease_labels).flatten())
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=disease_columns, yticklabels=disease_columns)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix for Disease Classification')
# plt.show()


# LIME Explanation

# Preprocess function for LIME
def preprocess_image_for_lime(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    image_tensor = (image_tensor - mean) / std
    return image_tensor

# Grad-CAM Function
def grad_cam(model, image_tensor, target_class):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.zero_grad()

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def save_activations_hook(module, input, output):
        activations.append(output.detach())

    # Access the last convolutional layer (modify as needed)
    last_conv_layer = model.densenet.features[-1]  # Change to actual last conv layer if different
    handle_gradients = last_conv_layer.register_backward_hook(save_gradients_hook)
    handle_activations = last_conv_layer.register_forward_hook(save_activations_hook)

    # Forward pass
    disease_pred, view_pred = model(image_tensor)
    target = disease_pred[0][target_class]
    target.backward()

    # Remove hooks
    handle_gradients.remove()
    handle_activations.remove()

    # Compute Grad-CAM heatmap
    gradients = gradients[0].cpu().numpy()
    activations = activations[0].cpu().numpy()[0]  # Squeeze batch dimension

    weights = np.mean(gradients, axis=(1, 2))  # Average gradients
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)  # Apply ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
    cam = np.uint8(255 * cam)

    return cam

# Visualization Function
def visualize_with_gradcam_lime(image_tensor, model, target_class):
    # Convert the tensor to a numpy image
    img = image_tensor.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # LIME Explanation
    def lime_forward(image_array):
        image_tensor = torch.tensor(image_array).permute(0, 3, 1, 2).float().to(device)
        image_tensor = preprocess_image_for_lime(image_tensor).to(device)
        with torch.no_grad():
            disease_pred, _ = model(image_tensor)  # Assume disease_pred is the first output
        return disease_pred.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        lime_forward,
        top_labels=1,
        hide_color=0,
        num_samples=1000  # Reduce for faster execution
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    # Generate Grad-CAM heatmap
    grad_cam_heatmap = grad_cam(model, image_tensor, target_class)

    # Resize Grad-CAM to match image size and overlay
    grad_cam_heatmap_resized = cv2.resize(grad_cam_heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap_overlay = cv2.applyColorMap(grad_cam_heatmap_resized, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap_overlay, 0.4, 0)

    # Display original image with LIME and Grad-CAM
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title("Original Image")

    ax[1].imshow(temp)
    ax[1].axis('off')
    ax[1].set_title("LIME Explanation")

    ax[2].imshow(overlayed_image)
    ax[2].axis('off')
    ax[2].set_title("Grad-CAM Overlay")

    plt.show()

# Example of using both Grad-CAM and LIME
for test_images, _, _ in test_loader:
    target_class = 0  # Choose the class to visualize
    visualize_with_gradcam_lime(test_images[0].to(device), model, target_class)
    break  # Visualize just one image


