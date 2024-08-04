from flask import Flask, request, jsonify, url_for
import face_recognition
import cv2
import numpy as np
import os
import tempfile
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_label(image, text, top_left, bottom_right, font, font_scale, text_color, bg_color, thickness, radius):
    overlay = image.copy()
    draw_rounded_rectangle(overlay, top_left, bottom_right, bg_color, thickness, radius)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = top_left[0] + (bottom_right[0] - top_left[0] - text_size[0]) // 2
    text_y = top_left[1] + (bottom_right[1] - top_left[1] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    unknown_faces_dir = data.get('unknown_faces_dir')
    known_faces_dir = data.get('known_faces_dir')

    if not unknown_faces_dir or not known_faces_dir:
        return jsonify({"error": "Both 'unknown_faces_dir' and 'known_faces_dir' are required."}), 400

    # Delete the 'static' directory if it exists
    static_dir = os.path.join(os.getcwd(), 'static')
    if os.path.exists(static_dir):
        shutil.rmtree(static_dir)

    # Recreate the 'static' directory
    os.makedirs(static_dir)

    unknown_faces = [os.path.join(unknown_faces_dir, f) for f in os.listdir(unknown_faces_dir) if os.path.isfile(os.path.join(unknown_faces_dir, f))]
    known_faces = [os.path.join(known_faces_dir, f) for f in os.listdir(known_faces_dir) if os.path.isfile(os.path.join(known_faces_dir, f))]

    if not unknown_faces:
        return jsonify({"error": "No group photos found in the specified directory."}), 400

    if not known_faces:
        return jsonify({"error": "No known faces found in the specified directory."}), 400

    known_face_encodings = []
    known_face_names = []

    for known_face in known_faces:
        try:
            known_image = face_recognition.load_image_file(known_face)
        except Exception as e:
            continue

        known_encodings = face_recognition.face_encodings(known_image)
        if len(known_encodings) == 0:
            continue

        known_face_encodings.append(known_encodings[0])
        name = os.path.splitext(os.path.basename(known_face))[0]
        known_face_names.append(name)

    response_links = []
    threshold = 0.6

    for unknown_face in unknown_faces:
        try:
            unknown_image = face_recognition.load_image_file(unknown_face)
        except Exception as e:
            continue

        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        if not face_encodings:
            continue

        face_names = []

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = "Unknown"

            if face_distances[best_match_index] < threshold:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            draw_rounded_rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2, 10)
            label_top_left = (left, bottom + 5)
            label_bottom_right = (right, bottom + 35)
            draw_label(unknown_image, name, label_top_left, label_bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 255), 2, 5)

        bgr_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

        temp_filename = os.path.join(static_dir, secure_filename(next(tempfile._get_candidate_names()) + ".jpg"))
        cv2.imwrite(temp_filename, bgr_image)
        response_links.append(url_for('static', filename=os.path.basename(temp_filename), _external=True))

    return jsonify({"image_links": response_links})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
