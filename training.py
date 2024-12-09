import mediapipe as mp
from stone import process
import os
import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mp_hands = mp.solutions.hands

updated_dataset_path = r'D:\archive\updated_dataset.csv'  
updated_dataset = pd.read_csv(updated_dataset_path)

skin_tone_data = updated_dataset.set_index('imageName')['skinColor'].to_dict()

skin_tone_mapping = {"fair": 0, "medium": 1, "dark": 2}
jewelry_mapping = {0: "Diamond", 1: "Platinum", 2: "Gold"}

def is_hand_present(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks: 
            return True
        return False

def load_images_with_labels(folder, orientation_label, skin_tone_data, skin_tone_mapping, image_size=(100, 100)):
    orientation_data = []  
    skin_tone_data_rgb = []  
    orientation_labels = []
    skin_tone_labels = []
    
    for filename in os.listdir(folder):  
        img_path = os.path.join(folder, filename)  
        img = cv2.imread(img_path)  
        if img is not None:  
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            gray_img = cv2.resize(gray_img, image_size)  
            gray_img_flattened = gray_img.flatten()  

            rgb_img = cv2.resize(img, image_size)  
            rgb_img_flattened = rgb_img.flatten()  

            skin_tone_label = skin_tone_mapping.get(skin_tone_data.get(filename), -1)
            if skin_tone_label != -1:  
                orientation_data.append(gray_img_flattened)
                skin_tone_data_rgb.append(rgb_img_flattened)
                orientation_labels.append(orientation_label)
                skin_tone_labels.append(skin_tone_label)
    return orientation_data, skin_tone_data_rgb, orientation_labels, skin_tone_labels

front_folder = r"D:\archive\front_palm"
back_folder = r"D:\archive\back_palm"
orientation_model_path = "palm_orientation_model.pkl"
skin_tone_model_path = "skin_tone_model.pkl"

if not os.path.exists(orientation_model_path) or not os.path.exists(skin_tone_model_path):
    print("Training models...")

    front_orient_data, front_skin_data, front_orient_labels, front_skin_labels = load_images_with_labels(
        front_folder, 1, skin_tone_data, skin_tone_mapping
    )
    back_orient_data, back_skin_data, back_orient_labels, back_skin_labels = load_images_with_labels(
        back_folder, 0, skin_tone_data, skin_tone_mapping
    )

    orientation_data = np.array(front_orient_data + back_orient_data)  
    skin_tone_data = np.array(front_skin_data + back_skin_data)  
    orientation_labels = np.array(front_orient_labels + back_orient_labels)
    skin_tone_labels = np.array(front_skin_labels + back_skin_labels)

    X_train_orient, X_test_orient, y_train_orient, y_test_orient = train_test_split(
        orientation_data, orientation_labels, test_size=0.2, random_state=42
    )

    orientation_model = RandomForestClassifier(n_estimators=100, random_state=42)
    orientation_model.fit(X_train_orient, y_train_orient)

    with open(orientation_model_path, 'wb') as f:
        pickle.dump(orientation_model, f)

    X_train_tone, X_test_tone, y_train_tone, y_test_tone = train_test_split(
        skin_tone_data, skin_tone_labels, test_size=0.2, random_state=42
    )

    skin_tone_model = RandomForestClassifier(n_estimators=100, random_state=42)
    skin_tone_model.fit(X_train_tone, y_train_tone)

    with open(skin_tone_model_path, 'wb') as f:
        pickle.dump(skin_tone_model, f)

    print("Models trained and saved wooo hooo .")

    print("Orientation Model Accuracy:", accuracy_score(y_test_orient, orientation_model.predict(X_test_orient)))
    print("Skin Tone Model Accuracy:", accuracy_score(y_test_tone, skin_tone_model.predict(X_test_tone)))
else:
    # print("model Gemos")
    print("Loading ModeLs...")
    with open(orientation_model_path, 'rb') as f:
        orientation_model = pickle.load(f)
    with open(skin_tone_model_path, 'rb') as f:
        skin_tone_model = pickle.load(f)

# Predict Function
def predict_image(model_orientation, model_skin_tone, image_path, image_size=(100, 100)):
    img = cv2.imread(image_path)
    if img is not None and is_hand_present(img):  
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_img = cv2.resize(gray_img, image_size)
        gray_img_flattened = gray_img.flatten()


        # rgb_img = cv2.resize(img, image_size)
        # rgb_img_flattened = rgb_img.flatten()


        rgb_img = cv2.resize(img, image_size)
        rgb_img_flattened = rgb_img.flatten()

        orientation_pred = model_orientation.predict([gray_img_flattened])[0]
        orientation = "Front Palm" if orientation_pred == 1 else "Back Palm"
        # skin_tone_pred = model_skin_tone.predict([rgb_img_flattened])[0]
        # skin_tone = {0: , 1: "Medium", 2: "Dark"}[skin_tone_pred]

        skin_tone_pred = model_skin_tone.predict([rgb_img_flattened])[0]
        skin_tone = {0: "Fair", 1: "Medium", 2: "Dark"}[skin_tone_pred]

        jewelry = jewelry_mapping[skin_tone_pred]

        return {"Orientation": orientation, "Skin Tone": skin_tone, "Jewelry": jewelry}
    return {"Error": "Hand not detected"}

new_image_path = r"D:\jwl\AI-project-jewllry-recommendation-system\back_palm\Hand_0000080.jpg"
result = predict_image(orientation_model, skin_tone_model, new_image_path)
print("Prediction Result:", result)



# from stone import process

# def classify_brightness(skin_tone_hex):
#     """
#     Classify skin tone into bright, medium, or dark based on hex color.
#     """
#     # Convert hex to RGB
#     skin_tone_rgb = tuple(int(skin_tone_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
#     r, g, b = skin_tone_rgb

#     # Calculate brightness using relative luminance formula
#     brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b

#     # Classify brightness
#     if brightness < 85:
#         return "Dark"
#     elif brightness < 170:
#         return "Medium"
#     else:
#         return "Bright"

#     result2 = process(new_image_path)


# Test on a new image
# new_image_path = r"D:\jwl\AI-project-jewllry-recommendation-system\back_palm\Hand_0000009.jpg"
# enhanced_result = predict_with_stone(orientation_model, skin_tone_model, new_image_path)

# print("Enhanced Prediction Result:", enhanced_result)

# from stone import process

# def classify_brightness_from_rgb(rgb):
#     r, g, b = rgb

#     # Calculate brightness using relative luminance formula
#     brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b

#     # Classify brightness
#     if brightness < 85:
#         return "Dark"
#     elif brightness < 170:
#         return "Medium"
#     else:
#         return "Bright"

# def classify_skin_tone_with_brightness_direct(image_path):

#     # Process the image using the Stone library
#     result2 = process(image_path)

#     # Check if the result contains face data
#     if "faces" in result2 and len(result2["faces"]) > 0:
#         face_data = result2["faces"][0]  # Access the first face
#         skin_tone = face_data.get("skin_tone", "N/A")  # Hex color of the skin tone
#         dominant_colors = face_data.get("dominant_colors", [])
#         if skin_tone != "N/A" and dominant_colors:
#             # Use the dominant color for RGB classification
#             dominant_rgb = tuple(int(dominant_colors[0]["color"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
#             brightness_category = classify_brightness_from_rgb(dominant_rgb)
#             return {
#                 "Skin Tone": skin_tone,
#                 "Brightness Category": brightness_category,
#                 "Dominant RGB": dominant_rgb,
#             }
#         else:
#             return {"Error": "Skin tone or dominant color data unavailable."}
#     else:
#         return {"Error": "No face data available."}
# print(classify_skin_tone_with_brightness_direct(new_image_path))