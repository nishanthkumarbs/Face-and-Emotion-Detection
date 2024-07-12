import cv2
from fer import FER

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the FER model
emotion_detector = FER()

# Create a named window and set it to fullscreen
cv2.namedWindow('Face and Expression Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Face and Expression Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces and predict expressions
    face_count = 0
    for (x, y, w, h) in faces:
        face_count += 1
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Detect the emotion
        emotion, score = emotion_detector.top_emotion(face)
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the emotion label
        if emotion:
            cv2.putText(frame, f'{emotion} ({score:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the number of faces detected
    cv2.putText(frame, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame with detected faces and expressions
    cv2.imshow('Face and Expression Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

