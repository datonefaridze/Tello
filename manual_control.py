# source https://github.com/damiafuentes/DJITelloPy
from tellopy import Tello
import cv2
import math
import threading
import time
import numpy as np
import sys
import av

tello = Tello()
tello.connect()

while True:
    # img = tello.get_video_stream()
    # print(img.read(1))
    # print(img)
    container = av.open(tello.get_video_stream())
    print("container: ", container)
    video_st = container.decode(video=10)
    for frame in video_st:
        frame = imgArray = np.asarray(frame.to_image()) 
        print(type(frame))
        print("video_st: ", frame)
        cv2.imshow("video_st", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.quit()
        break






# import tellopy


# tello = Tello()# tello = Tello()
# tello.connect()

# while True:
#     # img = tello.get_video_stream()
#     # print(img.read(1))
#     # print(img)
#     container = av.open(tello.get_video_stream())
#     video_st = container.decode(video=0)
#     for frame in video_st:
#         frame = imgArray = np.asarray(frame.to_image()) 
#         print(type(frame))
#         print("video_st: ", frame)
#         cv2.imshow("video_st", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         tello.quit()
#         break

# tello.connect()
# print("dato: ")
# print(tello.get_battery())

# tello.streamon()
# frame_read = tello.get_frame_read()
# tello.takeoff()
# tello = Tello()
# tello.connect()

# while True:
#     # img = tello.get_video_stream()
#     # print(img.read(1))
#     # print(img)
#     container = av.open(tello.get_video_stream())
#     video_st = container.decode(video=0)
#     for frame in video_st:
#         frame = imgArray = np.asarray(frame.to_image()) 
#         print(type(frame))
#         print("video_st: ", frame)
#         cv2.imshow("video_st", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         tello.quit()
#         break




# while True:

#     img = tello.get_frame_read().frame
#     img = cv2.resize(img ,(100, 100))
#     cv2.imshow("drone", img)

#     key = cv2.waitKey(1) & 0xff
#     if key == 27:  # ESC
#         break
#     elif key == ord('w'):
#         tello.move_forward(30)
#     elif key == ord('s'):
#         tello.move_back(30)
#     elif key == ord('a'):
#         tello.move_left(30)
#     elif key == ord('d'):
#         tello.move_right(30)
#     elif key == ord('e'):
#         tello.rotate_clockwise(30)
#     elif key == ord('q'):
#         tello.rotate_counter_clockwise(30)
#     elif key == ord('r'):
#         tello.move_up(30)
#     elif key == ord('f'):
#         tello.move_down(30)

# tello.land()
