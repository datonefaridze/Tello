import time
from time import sleep
import tellopy

drone = tellopy.Tello()
drone.connect()
# drone.takeoff()
# time.sleep(100)
drone.land()
drone.quit()
