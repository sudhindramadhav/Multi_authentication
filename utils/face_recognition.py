import cv2
import base64
import numpy as np
import dlib

# Initialize dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def capture_face_data():
    """Improved face capture with better feedback"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible!")
        return None

    print("🎥 Press 'C' to capture (make sure face is clearly visible), or 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame!")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use both Haar and dlib detectors for better reliability
        faces_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        faces_dlib = face_detector(frame, 1)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces_haar:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for face in faces_dlib:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Face Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces_haar) > 0 or len(faces_dlib) > 0:
                # Prefer dlib detection if available
                if len(faces_dlib) > 0:
                    face = faces_dlib[0]
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                else:
                    x, y, w, h = faces_haar[0]
                
                face_img = frame[y:y+h, x:x+w]
                
                # Ensure the face image is valid
                if face_img.size == 0:
                    print("⚠️ Captured face image is empty!")
                    continue
                    
                _, buffer = cv2.imencode('.jpg', face_img)
                face_data = base64.b64encode(buffer).decode('utf-8')
                
                cap.release()
                cv2.destroyAllWindows()
                return face_data
            else:
                print("⚠️ No face detected! Please position your face clearly in frame.")
                
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def encode_faces(image):
    """Encodes face using Dlib's face recognition model."""
    detected_faces = face_detector(image, 1)
    if len(detected_faces) == 0:
        print("⚠️ No face detected in the image!")
        return None

    shape = shape_predictor(image, detected_faces[0])
    face_encoding = np.array(face_recognizer.compute_face_descriptor(image, shape))
    
    return face_encoding

def calculate_similarity(stored_face_path, new_face_data):
    """Calculates similarity between stored and new face data."""

    # Decode new face image from base64
    try:
        new_face_image = cv2.imdecode(np.frombuffer(base64.b64decode(new_face_data), np.uint8), cv2.IMREAD_COLOR)
        if new_face_image is None:
            print("❌ Failed to decode the new face image.")
            return 0
    except Exception as e:
        print(f"❌ Error decoding base64 image: {e}")
        return 0

    # Load stored face image
    stored_face_image = cv2.imread(stored_face_path)
    if stored_face_image is None:
        print("❌ Failed to load stored face image.")
        return 0

    # Encode faces
    stored_face_encodings = encode_faces(stored_face_image)
    new_face_encodings = encode_faces(new_face_image)

    if stored_face_encodings is None or new_face_encodings is None:
        print("❌ Face encoding failed. Make sure faces are detected.")
        return 0

    print(f"📌 Stored Face Encodings: {stored_face_encodings[:5]} ...")  # Print first 5 elements
    print(f"📌 New Face Encodings: {new_face_encodings[:5]} ...")

    # Compute Euclidean distance
    distance = np.linalg.norm(stored_face_encodings - new_face_encodings)

    # Convert distance to similarity percentage
    similarity_percentage = max(0, (1 - (distance / 0.6)) * 100)  # 0.6 is an empirically chosen threshold

    print(f"✅ Similarity Percentage: {similarity_percentage:.2f}%")
    return similarity_percentage

# Main execution
if __name__ == "__main__":
    # Capture face data
    face_data = capture_face_data()
    
    if face_data:
        # Path to the stored face image
        stored_face_path = 'stored_face.jpg'
        
        # Calculate similarity
        similarity_percentage = calculate_similarity(stored_face_path, face_data)
        
        if similarity_percentage > 50:  # Adjust threshold as needed
            print("✅ Faces are similar!")
        else:
            print("❌ Faces are not similar!")
    else:
        print("No face data captured.")
