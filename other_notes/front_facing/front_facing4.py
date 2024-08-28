import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize mediapipe face detection and landmark detection modules
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load an image
image = cv2.imread("A000260.jpg")  # Replace with the path to your image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize face mesh and detect facial landmarks
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(image)
    annotated_image = image.copy()

    # Extract specific landmarks by their indices: 1, 113, and 342
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]  # Assuming only one face is detected
        landmark_points = [(landmarks.landmark[1].x * image.shape[1], landmarks.landmark[1].y * image.shape[0]),
                           (landmarks.landmark[113].x * image.shape[1], landmarks.landmark[113].y * image.shape[0]),
                           (landmarks.landmark[342].x * image.shape[1], landmarks.landmark[342].y * image.shape[0])]

        # Define the transformation matrices based on the points
        pts1 = np.float32(landmark_points)
        pts2 = np.float32([[0, 0], [500, 0], [0, 500]])  # Define the output positions for the points

        # Perform perspective transformation
        matrix = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(image, matrix, (2000, 2000))

        # Visualize the landmarks on the transformed image
        plt.figure(figsize=(8, 8))
        plt.imshow(result)
        plt.scatter(pts2[:, 0], pts2[:, 1], c='red', s=50)  # Plotting transformed points
        plt.title('Transformed Landmarks')
        plt.show()
