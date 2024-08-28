import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2

def extract_face_landmarks(face_mesh, image):
    # Convert image to RGB (MediaPipe requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face landmark detection
    results = face_mesh.process(image_rgb)
    row, column = image.shape[:2]

    face_landmarks = []
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            landmark_points = np.zeros([468, 3])  # 468 landmarks in FaceMesh
            for i, landmark in enumerate(face_landmark.landmark):
                landmark_points[i] = [landmark.x * row, landmark.y * column, landmark.z * row]
            face_landmarks.append(landmark_points)
    
    return face_landmarks

def calculate_transformation_matrices(face_landmarks):
    transformation_matrices = []
    for landmarks in face_landmarks:
        # Selecting specific landmark indices for reference points
        p0 = landmarks[1]   # Nose
        p1 = landmarks[113]  # Corner of left eye
        p2 = landmarks[342]  # Corner of right eye
        
        # Calculate vectors and axes based on selected landmarks
        vx = p1 - p0
        vd = p2 - p0
        vx = vx / np.linalg.norm(vx)
        vz = np.cross(vx, vd)
        vz = vz / np.linalg.norm(vz)
        vy = np.cross(vz, vx)
        
        # Construct the transformation matrix 'm'
        m = np.zeros([4, 4])
        m[0:3, 0] = vx
        m[0:3, 1] = vy
        m[0:3, 2] = vz
        m[0:3, 3] = p0
        m[3, 3] = 1
        
        # Append the transformation matrix 'm' to the list
        transformation_matrices.append(m)
        print(m)
        
    return transformation_matrices

def apply_perspective_transformation(image, transformations):
    for matrix in transformations:
        # Convert the 4x4 matrix to a 3x3 matrix
        matrix_3x3 = matrix[:3, :3]

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(image, matrix_3x3, (image.shape[1], image.shape[0]))

        # Display the transformed image
        plt.figure(figsize=(8, 8))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Show the original image
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))  # Show the transformed image
        plt.title('Transformed Face')
        plt.axis('off')

        plt.show()

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

image = cv2.imread('A000260.jpg')

face_landmarks = extract_face_landmarks(mp_face_mesh, image)
transformations = calculate_transformation_matrices(face_landmarks)

apply_perspective_transformation(image, transformations)

mp_face_mesh.close()
