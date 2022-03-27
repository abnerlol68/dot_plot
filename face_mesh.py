import cv2
import os
import json
import mediapipe as mp

def dot_plot():
    img_path = './AR face database/Man_original/'
    img_path_list = os.listdir(img_path)

    if not os.path.exists('./data/'):
        os.makedirs('./data/')
        print('Created folder: ./data/')

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    points_list = [70, 63,105, 66, 107, 336, 296, 334, 293, 300, 122, 196, 3, 51, 281, 248, 419, 351, 37, 0, 267]

    data = {}
    data_points = {}
    data_points_x = {}
    data_points_y = {}
    data_points_z = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        for img_name in img_path_list:
            print('Analizando {}/{}'.format(img_path,img_name))
            image = cv2.imread('{}/{}'.format(img_path,img_name))
            height, width, _ = image.shape

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            img_name_str = img_name[0:-4]

            if results.multi_face_landmarks is not None:
                for face_landmarks in results.multi_face_landmarks:
                    """ 21 puntos elegidos del rostro """
                    for point in points_list:
                        data_points[point] = {
                            "x": face_landmarks.landmark[point].x,
                            "y": face_landmarks.landmark[point].y,
                            "z": face_landmarks.landmark[point].z
                        }
            
            data[img_name_str] = data_points

        if not os.path.exists('./data/points.json'):
            print('Created file: ./data/points.json')
        file = open('./data/points.json', "w")
        json.dump(data, file)
        file.close()
