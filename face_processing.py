import cv2
import face_recognition
import numpy as np
from models import db, Student

# Load student face encodings
def load_known_faces():
    students = Student.query.all()
    known_face_encodings = []
    known_face_names = []

    for student in students:
        if student.face_encoding:
            encoding_array = np.array(student.face_encoding)
            known_face_encodings.append(encoding_array)
            known_face_names.append(student.name)

    return known_face_encodings, known_face_names

def encode_face(image_path):
    """Extract face encodings from an image file."""
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) == 0:
        return None

    return face_encodings[0]

def recognize_faces():
    known_face_encodings, known_face_names = load_known_faces()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            print(f"Detected: {name}")

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()