import numpy as np
import cv2
from pathlib import Path
import os
from keras import Model
from keras.preprocessing import image
from keras.models import load_model
import PIL.Image
from scipy.spatial import distance

custom_model = load_model("C:/Users/SAMSUNG/Downloads/model_epoch003_whole.h5")
layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=custom_model.input,
                                 outputs=custom_model.get_layer(layer_name).output)

data_predict = {}

root_folder = 'data'
face_cascade = cv2.CascadeClassifier(
    'C:/Users/SAMSUNG/PycharmProjects/untitled/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def find_files(catalog):
    dirs_path = list(Path(catalog).iterdir())

    for files_path in dirs_path:
        for img_path in files_path.iterdir():
            img = cv2.imread(str(img_path))
            if img is None:
                print(str(img_path))
                continue
                
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 2)

            for (x, y, w, h) in faces:
                x, y, w, h = faces[0]
                center = (x + w // 2, y + h // 2)
                faceROI = cropping_face(gray, faces[0])
                a = np.asarray(PIL.Image.open(img_path))
                face = cv2.resize(a, (224, 224))
                test_img = image.img_to_array(face) / 255
                test_img = test_img.reshape(1, 224, 224, 3)
                output = intermediate_layer_model.predict(test_img)
                data_predict[Path(files_path).name + '/' + Path(img_path).name] = output


def cropping_face(img, face):
    x, y, w, h = face
    ratio = 0.2
    lt_x = int(x - w * ratio)
    lt_y = int(y - h * ratio)
    rb_x = int(x + w * (1 + ratio))
    rb_y = int(y + h * (1 + ratio))
    # print(lt_x, lt_y, rb_x, rb_y)
    left = -min([lt_x, 0])
    top = -min([lt_y, 0])
    right = max([rb_x, img.shape[1]])
    bottom = max([rb_y, img.shape[0]])

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    new_lt_x = int(x - w * 0.2) + left
    new_lt_y = int(y - h * 0.2) + top
    new_rb_x = int(x + w * 1.2) + left
    new_rb_y = int(y + h * 1.2) + top

    faceROI = img[new_lt_y:new_rb_y, new_lt_x:new_rb_x]
    faceROI = cv2.resize(faceROI, (200, 200))
    return faceROI


find_files(root_folder)


def web_face_vector(face):
    img_web = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    gray_web = cv2.cvtColor(img_web, cv2.COLOR_RGB2GRAY)
    face_web = face_cascade.detectMultiScale(gray_web, 1.3, 2)
    answer = []

    for i in range(len(face_web)):
        x, y, w, h = face_web[i]
        center = (x + w // 2, y + h // 2)
        faceROI_web = cropping_face(img_web, face_web[i])
        faceROI_web = cv2.resize(faceROI_web, (224, 224))
        faceROI_web = image.img_to_array(faceROI_web) / 255
        faceROI_web = faceROI_web.reshape(1, 224, 224, 3)
        output_web = intermediate_layer_model.predict(faceROI_web)
        answer.append([output_web, x, y, w, h])
    return answer


def people_om_web_cam(vector, min_cos_distance=1, people=0):
    for i in data_predict:
        c_d = distance.cosine(data_predict[i][0], vector[0])
        if c_d < min_cos_distance:
            min_cos_distance = c_d
            people = os.path.dirname(i)
    return people


cap = cv2.VideoCapture(0)

while True:
    ret_web, img_web = cap.read()
    answer_web_face_vector = web_face_vector(img_web)
    if answer_web_face_vector != None:
        for i in range(len(answer_web_face_vector)):
            people = people_om_web_cam(answer_web_face_vector[i][0])
            x, y, w, h = answer_web_face_vector[i][1], answer_web_face_vector[i][2], answer_web_face_vector[i][3], answer_web_face_vector[i][4]
            cv2.rectangle(img_web, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_web, people, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("camera", img_web)
    else:
        people = None
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
cap.release()

cv2.destroyAllWindows()
