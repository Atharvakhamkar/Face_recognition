import cv2
import face_recognition


# # import os
# # import cv2
# # import face_recognition

# # Function to load images and labels from a directory
# def load_images_from_directory(directory):
#     images = []
#     labels = []
    
#     for filename in os.listdir(directory):
#         path = os.path.join(directory, filename)
        
#         if os.path.isfile(path):
#             image = face_recognition.load_image_file(path)
#             face_encoding = face_recognition.face_encodings(image)
            
#             if len(face_encoding) > 0:
#                 images.append(face_encoding[0])
#                 labels.append(directory)
#             else:
#                 print(f"No face found in {path}.")

#     return images, labels

# # Load known faces and their labels
# known_people_images, known_people_labels = load_images_from_directory("dataset/people")
# known_non_living_images, known_non_living_labels = load_images_from_directory("dataset/non-living")

# # Rest of your code remains the same...







# ******************************///////////////////////////////////////////////////*******************************

# # Load a sample picture and learn how to recognize it.
# known_image = face_recognition.load_image_file("parnav.jpg")
# known_face_encoding = face_recognition.face_encodings(known_image)[0]

# # Create arrays of known face encodings and their corresponding labels
# known_face_encodings = [known_face_encoding]
# # known_face_labels = ["Atharva","hrutu","pranav","paa","patil"]
# known_face_labels = ["parnav"]


#*****************************************************                               ************************************************
# List of known image paths

known_images = ["Akshay Kumar.jpg", "kay kay menon.jpeg", "Taapsee pannu.jpeg", "Rana daggubati.jpg", "Anupam Kher.jpg"]

# Initialize empty lists for known face encodings and labels
known_face_encodings = []
known_face_labels = []

# Load known faces and their labels
for image_path in known_images:
    known_image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(known_image)

    if len(face_encoding) > 0:
        known_face_encodings.append(face_encoding[0])
        known_face_labels.append(image_path.split('.')[0])  # Use the file name as the label
    else:
        print(f"No face found in {image_path}.")

# Initialize some variables
face_locations = []
face_encodings = []
face_labels = []
process_this_frame = True

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame to save time
    if process_this_frame:
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_labels = []
        for face_encoding in face_encodings:
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found, use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_labels[first_match_index]

            face_labels.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw the label below the face
        cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
