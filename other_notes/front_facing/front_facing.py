import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
            print(face_landmarks)
    
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
        cv2.imshow('Face', image)
        cv2.imshow('Transformed Face', transformed_image)
        print(transformed_image)
        cv2.waitKey(0)  # Wait for a key press to view the next transformation (or replace with appropriate wait time)

def refine_transformation_matrices(transformations, angles):
    refined_transformations = []
    for matrix in transformations:
        # Example angles for rotation along X, Y, Z axes
        angle_x, angle_y, angle_z = angles
        
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle_x)), -np.sin(np.radians(angle_x))],
            [0, np.sin(np.radians(angle_x)), np.cos(np.radians(angle_x))]
        ])

        rotation_matrix_y = np.array([
            [np.cos(np.radians(angle_y)), 0, np.sin(np.radians(angle_y))],
            [0, 1, 0],
            [-np.sin(np.radians(angle_y)), 0, np.cos(np.radians(angle_y))]
        ])

        rotation_matrix_z = np.array([
            [np.cos(np.radians(angle_z)), -np.sin(np.radians(angle_z)), 0],
            [np.sin(np.radians(angle_z)), np.cos(np.radians(angle_z)), 0],
            [0, 0, 1]
        ])

        # Combine rotation matrices along X, Y, Z axes
        combined_rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

        # Extract the 3x3 part of the transformation matrix
        matrix_3x3 = matrix[:3, :3]

        # Apply the combined rotation matrix to the 3x3 part of the transformation matrix
        refined_matrix_3x3 = np.dot(matrix_3x3, combined_rotation_matrix.T)  # Transpose for proper multiplication

        # Create a new 4x4 matrix with the refined 3x3 matrix and the translation component
        refined_matrix = np.eye(4)
        refined_matrix[:3, :3] = refined_matrix_3x3
        refined_matrix[:, 3] = matrix[:, 3]  # Preserve translation component

        refined_transformations.append(refined_matrix)

    return refined_transformations

#START
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

image = cv2.imread('.\A000260.jpg')

face_landmarks = extract_face_landmarks(mp_face_mesh, image)
transformations = calculate_transformation_matrices(face_landmarks)

# Assuming 'image' and 'transformations' are obtained from the previous code
image_shape = image.shape[:2]
angles = (0,0,0)
#refined_transformations = refine_transformation_matrices(transformations, angles)
refined_transformations = transformations
# Apply refined transformations
apply_perspective_transformation(image, refined_transformations)

# Further refining steps to iterate and adjust the transformation
# Here you might want to adjust the transformation matrices or use additional techniques for fine-tuning towards a front-facing view
# This could involve additional rotations, translations, or warping operations based on specific criteria

mp_face_mesh.close()