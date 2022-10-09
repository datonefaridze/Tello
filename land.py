import time
from time import sleep
import tellopy

drone = tellopy.Tello()
drone.connect()
drone.land()
drone.quit()
