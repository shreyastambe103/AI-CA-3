# Importing important libraries
import os
import tarfile
import shutil
import pandas as pd

# Step 1: Extracting images from .tar.gz files
for idx in range(1, 13):
    fn = 'images_%02d.tar.gz' % idx
    print(f'Extracting {fn}...')

    with tarfile.open(fn, 'r:gz') as tar:
        tar.extractall(path='extracted_images')  # Specifying a directory to extract images

print("Extraction complete.")

# Step 2: Organize images into train, val, test sets based on CSV files
# Defining the paths
extracted_dir = r"C:\Users\shreya\PycharmProjects\DL Project\extracted_images\images"
train_dir =r'C:\Users\shreya\PycharmProjects\DL Project\extracted_images\train'
val_dir = r'C:\Users\shreya\PycharmProjects\DL Project\extracted_images\val'
test_dir = r'C:\Users\shreya\PycharmProjects\DL Project\extracted_images\test'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Step 3: Load CSV files containing image paths and labels
train_df = pd.read_csv(r"C:\Users\shreya\Downloads\miccai2023_nih-cxr-lt_labels_train.csv")
val_df = pd.read_csv(r"C:\Users\shreya\Downloads\miccai2023_nih-cxr-lt_labels_val.csv")
test_df = pd.read_csv(r"C:\Users\shreya\Downloads\miccai2023_nih-cxr-lt_labels_test.csv")

# Step 4: Get list of image paths from each DataFrame and move the images
def move_images_from_csv(df, target_dir):
    for image_path in df['id']:
        # Creating full path to the image in the extracted folder
        full_image_path = os.path.join(extracted_dir, image_path)
        if os.path.exists(full_image_path):  # Makes sure the image exists
            shutil.move(full_image_path, target_dir)
        else:
            print(f"Image {full_image_path} not found.")

# Moves images to respective directories
move_images_from_csv(train_df, train_dir)
move_images_from_csv(val_df, val_dir)
move_images_from_csv(test_df, test_dir)

print("Images organized into train, val, and test sets based on CSV files.")


