import os
import pandas as pd
import shutil

# Load the CSV file
csv_file_path = r"D:\archive\HandInfo.csv"
hand_info_data = pd.read_csv(csv_file_path)

# Define the image folder and target folders
image_folder = r'D:\archive\palm_images'  # Replace with the actual path to your images
front_folder = './front_palm'  # Relative path for the front palm folder
back_folder = './back_palm'  # Relative path for the back palm folder

# Create target directories if they don't exist
os.makedirs(front_folder, exist_ok=True)
os.makedirs(back_folder, exist_ok=True)

# Iterate through the CSV and move images to respective folders
for _, row in hand_info_data.iterrows():
    image_name = row['imageName']
    aspect = row['aspectOfHand']
    
    # Determine source and destination
    src_path = os.path.join(image_folder, image_name)
    if 'palmar' in aspect:
        dest_path = os.path.join(front_folder, image_name)
    elif 'dorsal' in aspect:
        dest_path = os.path.join(back_folder, image_name)
    else:
        continue  # Skip if aspectOfHand is neither palmar nor dorsal
    
    # Move the image
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)

# Summary message
print(f"Images have been successfully separated into '{front_folder}' and '{back_folder}'.")
