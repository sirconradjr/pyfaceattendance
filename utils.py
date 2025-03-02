import os
import face_processing
import pandas as pd
from models import db, Student, Attendance

UPLOAD_FOLDER = "uploads"

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def save_student_image(image_file, student_name):
    """Save uploaded student image and return the file path."""
    filename = f"{student_name.replace(' ', '_')}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)
    return filepath


def encode_face(image_path):
    """Extract face encodings from an image file."""
    image = face_processing.load_image_file(image_path)
    face_encodings = face_processing.face_encodings(image)

    if len(face_encodings) == 0:
        return None  # No face detected

    return face_encodings[0]  # Return the first face encoding


def export_attendance_to_excel(section_id, date):
    """Export attendance data for a specific section and date as an Excel file."""
    records = Attendance.query.filter_by(section_id=section_id, date=date).all()

    if not records:
        return None  # No attendance records found

    data = [{
        "Student Name": Student.query.get(record.student_id).name,
        "Date": record.date.strftime("%Y-%m-%d"),
        "Status": record.status
    } for record in records]

    df = pd.DataFrame(data)
    file_path = f"static/attendance_{section_id}_{date}.xlsx"
    df.to_excel(file_path, index=False)

    return file_path  # Return the file path for downloading
