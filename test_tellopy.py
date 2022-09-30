# source https://github.com/damiafuentes/DJITelloPy
import tellopy
import cv2
import math
import threading
import time
import numpy as np
import sys
import av


def video_thread():
    global drone
    global run_video_thread
    global av
    print('START Video thread')
    drone.start_video()
    try:                                                                                                                            
        container = av.open(drone.get_video_stream())
        frame_count = 0
        while True:
            for frame in container.decode(video=0):
                frame_count = frame_count + 1
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                cv2.imshow('Original', image)
                cv2.waitKey(1)
        cv2.destroyWindow('Original')
    except KeyboardInterrupt as e:
        print('KEYBOARD INTERRUPT Video thread ' + e)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('EXCEPTION Video thread ' + e)

def main():    
    global drone
    drone = tellopy.Tello()
    drone.connect()
    
    try:
        threading.Thread(target=video_thread).start()
    except e:
        print(str(e))
    finally:
        print('Shutting down connection to drone...')        
        drone.quit()
        exit(1)

if __name__ == '__main__':
    main()