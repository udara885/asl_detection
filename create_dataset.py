# Import Libraries
import os
import pickle

import mediapipe as mp
import cv2

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a Hands object with specific configuration
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the dataset
DATA_DIR = 'Data'

# Lists to store data and corresponding labels
data = []
labels = []

# Update the classes to include 26 letters and 10 digits
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
          '20', '21', '22', '23', '24', '25','26', '27', '28', '29', '30', '31', '32', '33', '34', '35']

# Loop through each class label
for class_label in classes:
    class_dir = os.path.join(DATA_DIR, class_label)
    
    # Loop through each image in the class directory
    for img_path in os.listdir(class_dir):
        data_aux = []

        x_ = []
        y_ = []

        # Read the image and convert it to RGB
        img = cv2.imread(os.path.join(class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x, y coordinates of each hand landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates relative to the minimum x, y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the normalized hand landmark data and corresponding label
            data.append(data_aux)
            labels.append(class_label)

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)