import cv2
from simple_pid import PID
import cv2
import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
from custom_utils import *
import torch
from detector import Detector

width = 360
max_fb = 0
min_fb = 100000000
# 0.4, 0.4, 0
center_pid = PID(0.5, 0.1, 0.4, setpoint=0, output_limits=(-1, 1))
fb_range = [40000, 80000]
area_dist = {'area': []}

def track_face(areas, centers):
    print("areas: ", areas)
    print("centers: ", centers)
    if areas == [] and centers == []:
        return 0, 0
    area, bbox = areas[0]
    center, bbox = centers[0]
    print("track face")
    print("center: ", center)
    print("area: ", area)
    x,y = center
    print("x: ", x)
    diff = (x - width//2) / (width//2)
    yaw = center_pid(diff)
    fb = 0
    if x==0: return 0, 0
    if area >= fb_range[0] and area <= fb_range[1]:
        fb = 0
    if area >= fb_range[1]:
        fb=-30
    if area<=fb_range[0] and area!=0:
        fb=30   

    return fb, -yaw
    #me.send_rc_control(0, fb, 0, speed)

model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)

def main():
    drone = tellopy.Tello()
    first_frame=True
    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        drone.takeoff()
        drone.up(30)
        time.sleep(10)
        drone.up(0)
        
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                img = cv2.resize(image, (360, 360))

                # areas, centers = detect_haarcascade(img)
                areas, centers = detect_yolo(model, img)
                # import pdb; pdb.set_trace()
                if first_frame:
                    rect = cv2.selectROI("select the area", img)
                    first_frame = False
                    print(rect)
                    x1,y1,x2, y2=rect[0], rect[1], rect[2], rect[3]
    
                    img_crop=img[y1:y1+y2,x1:x1+x2]
                    show(img_crop)
                    detector = Detector(img_crop)
                    first_frame = False
                _, argmax, _ = detector(img, centers)
                print("argmax: ", argmax)
                print("centers: ", centers)

                if argmax!=None:
                    centers = [centers[argmax]]
                img = draw_figures(img, centers)

                fb, yaw = track_face(areas, centers)
                drone.set_yaw(yaw)
                if fb >0:
                    drone.forward(fb)
                if fb<=0:
                    drone.backward(-fb)
                print('yaw: ', yaw)
                area_dist['area'].append(areas)
                cv2.imshow('Original', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    drone.land()
                    drone.quit()
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
                    
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.land()
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
