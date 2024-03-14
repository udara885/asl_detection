# Import Libraries
import os
import cv2

# Define the directory to store the collected data
DATA_DIR = 'Data'

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and dataset size
number_of_classes = 1  # 26 letters + 10 digits
dataset_size = 100

# Open a connection to the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Loop through each class
for j in range(number_of_classes):
    # Create a subdirectory for each class
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Display a message to prompt the user to press "Q" to start capturing data for the current class
    done = False
    while True:
        ret, frame = cap.read()
        text = 'Press "Q" to capture class {}!'.format(j)
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Break the loop when the user presses 'Q'
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture and save images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        
        # Save the captured frame as an image in the corresponding class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()