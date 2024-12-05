import os
import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset for skin tone mapping
updated_dataset_path = r'D:\archive\updated_dataset.csv'  # Replace with your path
updated_dataset = pd.read_csv(updated_dataset_path)

# Create a mapping of image names to skin tones
skin_tone_data = updated_dataset.set_index('imageName')['skinColor'].to_dict()

# Define mappings for skin tones and jewelry preferences
skin_tone_mapping = {"fair": 0, "medium": 1, "dark": 2}
jewelry_mapping = {0: "Diamond", 1: "Platinum", 2: "Gold"}

# Function to load images and assign labels for palm orientation and skin tone
def load_images_with_labels(folder, orientation_label, skin_tone_data, skin_tone_mapping, image_size=(100, 100)):
    """
    Load images from a folder, resize them, and assign both orientation and skin tone labels.
    """
    data = []
    orientation_labels = []
    skin_tone_labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img_flattened = img.flatten()

            # Get skin tone label
            skin_tone_label = skin_tone_mapping.get(skin_tone_data.get(filename), -1)
            if skin_tone_label != -1:
                data.append(img_flattened)
                orientation_labels.append(orientation_label)
                skin_tone_labels.append(skin_tone_label)
    return data, orientation_labels, skin_tone_labels

# Define folders
front_folder = r"D:\archive\front_palm"
back_folder = r"D:\archive\back_palm"
orientation_model_path = "palm_orientation_model.pkl"
skin_tone_model_path = "skin_tone_model.pkl"

# Train models
if not os.path.exists(orientation_model_path) or not os.path.exists(skin_tone_model_path):
    print("Training models...")

    # Load images for front and back palms
    front_data, front_orient_labels, front_skin_labels = load_images_with_labels(
        front_folder, 1, skin_tone_data, skin_tone_mapping
    )
    back_data, back_orient_labels, back_skin_labels = load_images_with_labels(
        back_folder, 0, skin_tone_data, skin_tone_mapping
    )

    # Combine data and labels
    data = np.array(front_data + back_data)
    orientation_labels = np.array(front_orient_labels + back_orient_labels)
    skin_tone_labels = np.array(front_skin_labels + back_skin_labels)

    # Train-test split for orientation
    X_train_orient, X_test_orient, y_train_orient, y_test_orient = train_test_split(
        data, orientation_labels, test_size=0.2, random_state=42
    )

    # Train orientation model
    orientation_model = RandomForestClassifier(n_estimators=100, random_state=42)
    orientation_model.fit(X_train_orient, y_train_orient)

    # Save orientation model
    with open(orientation_model_path, 'wb') as f:
        pickle.dump(orientation_model, f)
    # Train-test split for skin tone
    X_train_tone, X_test_tone, y_train_tone, y_test_tone = train_test_split(
        data, skin_tone_labels, test_size=0.2, random_state=42
    )

    # Train skin tone model
    skin_tone_model = RandomForestClassifier(n_estimators=100, random_state=42)
    skin_tone_model.fit(X_train_tone, y_train_tone)

    # Save skin tone model
    with open(skin_tone_model_path, 'wb') as f:
        pickle.dump(skin_tone_model, f)

    print("Models trained and saved.")

    # Evaluate models
    print("Orientation Classification Report:")
    print(classification_report(y_test_orient, orientation_model.predict(X_test_orient)))

    print("Skin Tone Classification Report:")
    print(classification_report(y_test_tone, skin_tone_model.predict(X_test_tone)))
else:
    print("Loading models...")
    with open(orientation_model_path, 'rb') as f:
        orientation_model = pickle.load(f)
    with open(skin_tone_model_path, 'rb') as f:
        skin_tone_model = pickle.load(f)

# Predict Function
def predict_image(model_orientation, model_skin_tone, image_path, image_size=(100, 100)):
    """
    Predict palm orientation, skin tone, and jewelry preference for a given image.
    """
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, image_size)
        img_flattened = img.flatten()
        
        # Predict orientation
        orientation_pred = model_orientation.predict([img_flattened])[0]
        orientation = "Front Palm" if orientation_pred == 1 else "Back Palm"

        # Predict skin tone
        skin_tone_pred = model_skin_tone.predict([img_flattened])[0]
        skin_tone = {0: "Fair", 1: "Medium", 2: "Dark"}[skin_tone_pred]

        # Determine jewelry preference
        jewelry = jewelry_mapping[skin_tone_pred]

        return {"Orientation": orientation, "Skin Tone": skin_tone, "Jewelry": jewelry}
    return {"Error": "Invalid Image"}

# Test on a new image
new_image_path = r"D:\archive\back_palm\Hand_0000008.jpg"
result = predict_image(orientation_model, skin_tone_model, new_image_path)
print("Prediction Result:", result)
