from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
me.takeoff()
print(me.get_battery())
me.streamon()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break