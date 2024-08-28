import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2

# Function to extract face landmarks using MediaPipe
def extract_face_landmarks(face_mesh, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    row, column = image.shape[:2]

    face_landmarks = []
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            landmark_points = np.zeros([468, 3])
            for i, landmark in enumerate(face_landmark.landmark):
                landmark_points[i] = [landmark.x * column, landmark.y * row, landmark.z]
            face_landmarks.append(landmark_points)
    
    return face_landmarks

# Function to visualize landmarks on the image
def visualize_landmarks(image, landmarks):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for landmark in landmarks:
        plt.scatter(landmark[:, 0], landmark[:, 1], s=5, marker='.', c='red')
    plt.title('Image with Landmarks')
    plt.axis('off')
    plt.show()

# Function to calculate transformation matrices for perspective correction
def calculate_transformation_matrices(face_landmarks):
    transformation_matrices = []
    for landmarks in face_landmarks:
        p0 = landmarks[113]
        p1 = landmarks[1]
        p2 = landmarks[342]
        
        vx = p1 - p0
        vd = p2 - p0
        vx = vx / np.linalg.norm(vx)
        vz = np.cross(vx, vd)
        vz = vz / np.linalg.norm(vz)
        vy = np.cross(vz, vx)
        
        m = np.zeros([4, 4])
        m[0:3, 0] = vx
        m[0:3, 1] = vy
        m[0:3, 2] = vz
        m[0:3, 3] = p0
        m[3, 3] = 1
        
        transformation_matrices.append(m)
        
    return transformation_matrices

# Function to apply perspective transformation to the image and landmarks
def apply_perspective_transformation(image, transformations, face_landmarks):
    for matrix, landmarks in zip(transformations, face_landmarks):
        matrix_3x3 = matrix[:3, :3]
        transformed_image = cv2.warpPerspective(image, matrix_3x3, (image.shape[1], image.shape[0]))

        landmark_points = landmarks[:, :2]
        transformed_landmarks = cv2.perspectiveTransform(np.array([landmark_points]), matrix_3x3)
        transformed_landmarks = transformed_landmarks[0]

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        plt.scatter(transformed_landmarks[:, 0], transformed_landmarks[:, 1], s=10, marker='.', c='blue')
        plt.title('Transformed Face with Landmarks')
        plt.axis('off')

        plt.show()


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

image = cv2.imread('A000001.jpg')
#image = cv2.imread('A000260.jpg')

face_landmarks = extract_face_landmarks(mp_face_mesh, image)
visualize_landmarks(image, face_landmarks)
transformations = calculate_transformation_matrices(face_landmarks)
apply_perspective_transformation(image, transformations, face_landmarks)

mp_face_mesh.close()
