import time
from time import sleep
import tellopy

drone = tellopy.Tello()
drone.connect()
drone.takeoff()
# time.sleep(100)

print("drone.set_yaw(0.3):")

drone.set_yaw(0.3)
time.sleep(5)
print("drone.set_yaw(-0.3):")
drone.set_yaw(-0.3)

time.sleep(5)
print("drone.set_yaw(0):")
drone.set_yaw(0)




