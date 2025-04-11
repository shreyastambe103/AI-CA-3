'''To handle Class Imbalance we create class weights so that the model pays more attention to classes that exist less number of times comparatively'''

# Importing important libraries
import pandas as pd
import torch
import torch.nn as nn

# Loading the train dataset
df = pd.read_csv(r'C:\Users\shreya\PycharmProjects\DL Project\train_final.csv')

# Calculating class weights based on existing binary columns for diseases
disease_columns = list(df.columns[1:21])
class_counts = df[disease_columns].sum().to_dict()
print("Class Counts per Disease:", class_counts)
total_samples = len(df)
print("Total Samples:", total_samples)

# Calculate class weights
total_samples=total_samples/20 #for 20 classes
class_weights = {label: total_samples / count for label, count in class_counts.items()}

# Convert class weights to tensor for PyTorch
class_weights_tensor = torch.FloatTensor(list(class_weights.values()))

# Normalize the class weights to keep them in a manageable range
normalized_class_weights_tensor = class_weights_tensor / class_weights_tensor.max()

# Print the class weights for reference
print("Class Weights:", class_weights)
print("Normalized Class Weights Tensor:", normalized_class_weights_tensor)

# Save the class weights tensor
torch.save(normalized_class_weights_tensor, 'class_weights_tensor.pth')

# Step 2: Define the loss function with normalized class weights
criterion = nn.BCEWithLogitsLoss(weight=normalized_class_weights_tensor)

# # Step 3: Example of how you can use the criterion during training
# outputs = model(inputs)  # Your model forward pass
# loss = criterion(outputs, labels)  # Calculate loss with class weights
