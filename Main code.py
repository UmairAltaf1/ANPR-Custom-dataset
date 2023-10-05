from tensorflow import keras
import numpy as np
import cv2
import pytesseract
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

model = keras.models.load_model('firsttarinedmodel.h5', compile=False)

# Define the label names
labels = ['positive', 'negative']

# Load the camera
cap = cv2.VideoCapture(1)

# Configure pytesseract to use the appropriate language and path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'

# Create a window to display the frame
cv2.namedWindow('Real-time Detection')

# Create a trackbar to adjust the threshold for better OCR results
initial_threshold = 127
cv2.createTrackbar('Threshold', 'Real-time Detection', initial_threshold, 255, lambda x: None)

# Initialize Firebase
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Read the OCR result from the Excel file
ocr_data = pd.read_excel('ocr_result.xlsx')
ocr_records = ocr_data['Text'].values.tolist()

for record in ocr_records:
    # Save the record to Firebase
    doc_ref = db.collection('ocr_records').document()
    doc_ref.set({
        'text': record
    })
    print('Record saved to Firebase:', record)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the image
    resized_frame = cv2.resize(frame, (256, 256))
    scaled_frame = resized_frame / 255.0
    input_frame = np.expand_dims(scaled_frame, axis=0)

    # Make predictions using the model
    predictions = model.predict(input_frame)[0]
    label_index = np.argmax(predictions)
    label = labels[label_index]
    confidence = predictions[label_index]

    # Draw a bounding box and label on the frame
    if confidence > 0.5 and label == 'positive':
        # Find contours in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, thresholded_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Get the threshold value from the trackbar
        threshold = cv2.getTrackbarPos('Threshold', 'Real-time Detection')

        # Apply the threshold value
        _, thresholded_frame = cv2.threshold(blurred_frame, threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assuming it corresponds to the car number plate)
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the bounding box around the number plate
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Apply OCR to the number plate region
        number_plate_region = gray_frame[y:y + h, x:x + w]
        text = pytesseract.image_to_string(number_plate_region, config=tessdata_dir_config)

        # Display the OCR result
        cv2.putText(frame, text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the OCR result to the Excel file when "s" key is pressed
        if cv2.waitKey(1) == ord('s'):
            ocr_data = ocr_data.append({'Text': text}, ignore_index=True)
            ocr_data.to_excel('ocr_result.xlsx', index=False)
            print('OCR result saved to ocr_result.xlsx')

            # Save the new record to Firebase
            doc_ref = db.collection('ocr_records').document()
            doc_ref.set({
                'text': text
            })
            print('Record saved to Firebase:', text)

    # Display the frame
    cv2.imshow('Real-time Detection', frame)

    # Quit the program if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
