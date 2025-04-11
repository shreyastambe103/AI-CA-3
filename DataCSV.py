# Importing important libraries
import pandas as pd
import os

# Load the CSV files
train_df = pd.read_csv(r"C:\Users\shreya\Downloads\miccai2023_nih-cxr-lt_labels_train.csv")
val_df = pd.read_csv(r"C:\Users\shreya\Downloads\miccai2023_nih-cxr-lt_labels_val.csv")
test_df = pd.read_csv(r"C:\Users\shreya\Downloads\miccai2023_nih-cxr-lt_labels_test.csv")
data_entry_df = pd.read_csv(r"C:\Users\shreya\Downloads\Data_Entry_2017_v2020.csv")
bbox_df = pd.read_csv(r"C:\Users\shreya\Downloads\BBox_List_2017.csv")

# Pre-processing files
bbox_df= bbox_df.iloc[:,:-3]
bbox_df=bbox_df.drop(columns=["Finding Label"])

train_df.rename(columns={'id':'Image Index'}, inplace=True)
val_df.rename(columns={'id':'Image Index'}, inplace=True)
test_df.rename(columns={'id':'Image Index'}, inplace=True)

# Merging train, val, and test data with data_entry to include patient info and view positions
train_merged = train_df.merge(data_entry_df, on='Image Index', how='left')
val_merged = val_df.merge(data_entry_df, on='Image Index', how='left')
test_merged = test_df.merge(data_entry_df, on='Image Index', how='left')

# Now merge with bounding boxes for those images that have bounding box information
train_final = train_merged.merge(bbox_df, on='Image Index', how='left')
val_final = val_merged.merge(bbox_df, on='Image Index', how='left')
test_final = test_merged.merge(bbox_df, on='Image Index', how='left')

# Preprocessing merged files
# Handling missing values (Example: Fill 0 in bounding box with no values)
train_final[['Bbox [x', 'y', 'w', 'h]']] = train_final[['Bbox [x', 'y', 'w', 'h]']].fillna(0)
val_final[['Bbox [x', 'y', 'w', 'h]']] = val_final[['Bbox [x', 'y', 'w', 'h]']].fillna(0)
test_final[['Bbox [x', 'y', 'w', 'h]']] = test_final[['Bbox [x', 'y', 'w', 'h]']].fillna(0)


# Removing rows(image info) for which images do no exist in the dataset
train_final = train_final[train_final.apply(lambda x: os.path.exists(os.path.join(r"C:\Users\shreya\PycharmProjects\DL Project\extracted_images\train", x[0])), axis=1)]
val_final = val_final[val_final.apply(lambda x: os.path.exists(os.path.join(r"C:\Users\shreya\PycharmProjects\DL Project\extracted_images\val", x[0])), axis=1)]
test_final = test_final[test_final.apply(lambda x: os.path.exists(os.path.join(r"C:\Users\shreya\PycharmProjects\DL Project\extracted_images\test", x[0])), axis=1)]

# See the final merged DataFrame
print("Train Dataset : \n")
print(train_final.shape)
print(train_final.head())
print("Validation Dataset : \n")
print(val_final.shape)
print(val_final.head())
print("Test Dataset : \n")
print(test_final.shape)
print(test_final.head())

# Saving the final merged datasets
train_final.to_csv('train_final.csv', index=False)
val_final.to_csv('val_final.csv', index=False)
test_final.to_csv('test_final.csv', index=False)
