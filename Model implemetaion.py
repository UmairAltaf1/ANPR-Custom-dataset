import cv2
import numpy as np
import tensorflow as tf

# Load the h5 model
model = tf.keras.models.load_model('model.h5')

# Define the classes
classes = ['class_1', 'class_2']

# Load the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32)
    frame /= 255.0
    
    # Make a prediction
    output = model.predict(np.array([frame]))
    
    # Extract the bounding boxes and labels from the output
    boxes = output[0][:, :4]
    scores = output[0][:, 4:]
    labels = np.argmax(scores, axis=1)
    
    # Visualize the results
    for box, label in zip(boxes, labels):
        if scores[0][label] > 0.5:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, classes[label], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
