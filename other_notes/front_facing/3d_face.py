import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scatter(face_landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for landmarks in face_landmarks:
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        z = landmarks[:, 2]

        ax.scatter(x, y, z, c='r', marker='o')  # Scatter plot for each face

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


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

#START
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

image = cv2.imread('.\A000001.jpg')

face_landmarks = extract_face_landmarks(mp_face_mesh, image)
plot_3d_scatter(face_landmarks)

mp_face_mesh.close()