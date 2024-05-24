import face_recognition
import cv2
import numpy as np
import os

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius):
    """Draw a rounded rectangle."""
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
    """Draw a label with text."""
    overlay = image.copy()
    draw_rounded_rectangle(overlay, top_left, bottom_right, bg_color, thickness, radius)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = top_left[0] + (bottom_right[0] - top_left[0] - text_size[0]) // 2
    text_y = top_left[1] + (bottom_right[1] - top_left[1] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

# Paths to the images
unknown_image_path = "assets/images/S3.jpeg"
known_image_paths = ["assets/images/Mike.jpeg", "assets/images/Dustin.jpeg", "assets/images/Eleven.jpeg", "assets/images/Lucas.jpeg", "assets/images/Max.jpeg", "assets/images/Will.jpeg"]

# Load the unknown image
try:
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    print(f"Successfully loaded unknown image from {unknown_image_path}")
except Exception as e:
    print(f"Failed to load unknown image: {e}")
    exit()

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

print(f"Found {len(face_encodings)} face(s) in the unknown image.")

if not face_encodings:
    print("No faces found in the unknown image.")
    exit()

# Load known images and get face encodings
known_face_encodings = []
known_face_names = []

for known_image_path in known_image_paths:
    try:
        known_image = face_recognition.load_image_file(known_image_path)
        print(f"Successfully loaded known image from {known_image_path}")
    except Exception as e:
        print(f"Failed to load known image: {e}")
        continue
    
    # Get the face encodings
    known_encodings = face_recognition.face_encodings(known_image)
    if len(known_encodings) == 0:
        print(f"No faces found in the known image {known_image_path}.")
        continue
    
    known_face_encodings.append(known_encodings[0])
    
    # Extract the name from the file path
    name = os.path.splitext(os.path.basename(known_image_path))[0]
    known_face_names.append(name)

print("Known face encodings loaded successfully.")

# Initialize some variables
face_names = []

# Set a lower threshold for face distance
threshold = 0.6  # Adjusted threshold for better results

# Process each face found in the unknown image
for face_encoding in face_encodings:
    # Use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    name = "Unknown"

    # Print the face distances for debugging
    print(f"Face distances: {face_distances}")

    # Check if the best match is within the threshold
    if face_distances[best_match_index] < threshold:
        name = known_face_names[best_match_index]

    face_names.append(name)
    print(f"Match found: {name} with distance: {face_distances[best_match_index]}")

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Draw a rounded rectangle around the face
    draw_rounded_rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2, 10)

    # Draw a label with a name below the face
    label_top_left = (left, bottom + 5)
    label_bottom_right = (right, bottom + 35)
    draw_label(unknown_image, name, label_top_left, label_bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 255), 2, 5)

# Convert the image from RGB (face_recognition uses RGB) to BGR (OpenCV uses BGR)
bgr_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Display the resulting image
cv2.imshow('Image', bgr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
