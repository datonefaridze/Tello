from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())