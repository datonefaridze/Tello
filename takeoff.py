import time
from time import sleep
import tellopy

drone = tellopy.Tello()
drone.connect()
drone.takeoff()
drone.up(30)
time.sleep(10)
drone.up(0)

time.sleep(10)
drone.land()
drone.quit()




