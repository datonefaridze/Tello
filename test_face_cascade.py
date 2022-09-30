import cv2
from simple_pid import PID
import pandas as pd
from djitellopy import tello
import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
width = 360
max_fb = 0
min_fb = 100000000
center_pid = PID(1, 0.1, 0.05, setpoint=width//2, output_limits=(-100, 100))
fb_range = [10000, 60000]
area_dist = {'area': []}

def track_face(me, info):
    area = info[0]
    x, y = info[1]
    print("x: ", x)
    speed = center_pid(x)
    fb = 0
    if x==0: return 0, 0
    if area >= fb_range[0] and area <= fb_range[1]:
        fb = 0
    if area >= fb_range[1]:
        fb=-20
    if area<=fb_range[0] and area!=0:
        fb=20   

    return fb, -speed
    #me.send_rc_control(0, fb, 0, speed)


def detect_face(faces):
    face_areas = []
    face_centers = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        face_areas.append(area)
        face_centers.append([cx, cy])

    face_area = 0
    face_center = (0,0)
    if len(face_areas) !=0:
        max_idx = face_areas.index(max(face_areas))
        face_area = face_areas[max_idx]
        face_center = face_centers[max_idx]
    return face_area, face_center



me = tello.Tello()
me.connect()
print("get_battery:", me.get_battery())
me.streamon()
me.takeoff()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_area, face_center = detect_face(faces)
    area_dist['area'].append(face_area)
    print("face_area: ", face_area)
    print("face_center: ", face_center)
    cv2.circle(img, face_center, 5, (0,255,255), cv2.FILLED)
    fb, speed = track_face(me=1, info=(face_area, face_center))
    print('fb: ', fb)
    print('speed: ', speed)
    min_fb = min(face_area, min_fb)
    max_fb = max(face_area, max_fb)
    # df = pd.DataFrame(area_dist)
    # df.to_csv("dato.csv")
    # Display
    print("min_fb: ", min_fb)
    print("max_fb: ", max_fb)
    cv2.imshow('img', img)
    me.send_rc_control(0,0,0,int(speed))
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# cap = cv2.VideoCapture(0)

# while True:
#     _, img = cap.read()
#     img = cv2.resize(img, (360, 240))

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     face_area, face_center = detect_face(faces)
#     area_dist['area'].append(face_area)
#     print("face_area: ", face_area)
#     print("face_center: ", face_center)
#     cv2.circle(img, face_center, 5, (0,255,255), cv2.FILLED)
#     fb, speed = track_face(me=1, info=(face_area, face_center))
#     print('fb: ', fb)
#     print('speed: ', speed)
#     min_fb = min(face_area, min_fb)
#     max_fb = max(face_area, max_fb)
#     df = pd.DataFrame(area_dist)
#     df.to_csv("dato.csv")
#     # Display
#     print("min_fb: ", min_fb)
#     print("max_fb: ", max_fb)
#     cv2.imshow('img', img)
#     # Stop if escape key is pressed

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # Release the VideoCapture object
# cap.release()