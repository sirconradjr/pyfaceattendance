from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import pandas as pd
import face_recognition
import numpy as np
from models import db, init_db, Section, Student, Attendance
from datetime import datetime
from utils import save_student_image, encode_face
from face_processing import load_known_faces, encode_face
import threading

stop_attendance = False

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///attendance.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

init_db(app)

# Route: Home
@app.route("/")
def index():
    return render_template("index.html")

# Route: Admin
@app.route('/admin')
def admin():
    return render_template('admin.html')

# Route: Create Section
@app.route("/create_section", methods=["POST"])
def create_section():
    data = request.json
    new_section = Section(name=data["name"])
    db.session.add(new_section)
    db.session.commit()

    sections = Section.query.all()
    return jsonify([
        {"id": s.id, "name": s.name} for s in sections
    ])

# Route: Get Sections
@app.route("/get_sections")
def get_sections():
    sections = Section.query.all()
    return jsonify([{"id": s.id, "name": s.name} for s in sections])

# Route: Get Attendance
@app.route("/get_attendance/<int:section_id>/<date>", methods=["GET"])
def get_attendance(section_id, date):
    """Fetch attendance records for a specific section and date."""
    records = Attendance.query.filter_by(section_id=section_id, date=date).all()

    if not records:
        return jsonify({"error": "No attendance records found!"}), 404

    data = [{
        "student_name": Student.query.get(record.student_id).name,
        "date": record.date.strftime("%Y-%m-%d"),
        "status": record.status
    } for record in records]

    return jsonify(data)

# Route: Export Attendance
@app.route("/export_attendance/<int:section_id>/<date>")
def export_attendance(section_id, date):
    records = Attendance.query.filter_by(section_id=section_id, date=date).all()
    
    if not records:
        return jsonify({"error": "No attendance records found!"}), 404

    data = [{
        "Student Name": Student.query.get(record.student_id).name,
        "Date": record.date.strftime("%Y-%m-%d"),
        "Status": record.status
    } for record in records]

    df = pd.DataFrame(data)
    file_path = f"static/attendance_{section_id}_{date}.xlsx"
    df.to_excel(file_path, index=False)

    return file_path

# Route: Register Student
@app.route("/register_student", methods=["POST"])
def register_student():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        name = request.form.get("name")
        section_id = request.form.get("section_id")
        image = request.files["image"]

        if not name or not section_id:
            return jsonify({"error": "Missing student name or section"}), 400

        # Save image and process face encoding
        image_path = save_student_image(image, name)
        face_encoding = encode_face(image_path)

        if face_encoding is None:
            return jsonify({"error": "No face detected in the image"}), 400

        # Save student to database
        new_student = Student(name=name, section_id=section_id, face_encoding=face_encoding.tolist())
        db.session.add(new_student)
        db.session.commit()

        return jsonify({"message": f"Student '{name}' registered successfully!"}), 201
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

# Route: Record Attendance
@app.route("/record_attendance", methods=["POST"])
def record_attendance():
    global stop_attendance
    stop_attendance = False

    data = request.json
    section_id = data["section_id"]
    today_date = datetime.utcnow().date()

    video_capture = cv2.VideoCapture(0)
    known_face_encodings, known_face_names = load_known_faces()

    def process_frames():
        global stop_attendance
        with app.app_context():
            while not stop_attendance:
                ret, frame = video_capture.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if matches:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                            student = Student.query.filter_by(name=name).first()
                            if student:
                                existing_record = Attendance.query.filter_by(
                                    student_id=student.id,
                                    section_id=section_id,
                                    date=today_date
                                ).first()

                                if not existing_record:
                                    new_attendance = Attendance(
                                        student_id=student.id, 
                                        section_id=section_id, 
                                        date=today_date, 
                                        status="Present"
                                    )
                                    db.session.add(new_attendance)
                                    db.session.commit()
                                    print(f"✅ Attendance marked for {name}")

                cv2.imshow("Attendance System", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_attendance = True

            video_capture.release()
            cv2.destroyAllWindows()

    threading.Thread(target=process_frames, daemon=True).start()
    return jsonify({"message": "Attendance recording started!"})

# Route: Stop Attendance
@app.route("/stop_attendance", methods=["POST"])
def stop_attendance_route():
    global stop_attendance
    stop_attendance = True
    return jsonify({"message": "Attendance recording stopped!"})

if __name__ == "__main__":
    app.run(debug=True)
