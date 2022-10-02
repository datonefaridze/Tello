import time
from time import sleep
import tellopy

drone = tellopy.Tello()
drone.connect()
drone.takeoff()
# time.sleep()
drone.up(30)
time.sleep(10)
drone.up(0)

time.sleep(10)
drone.land()
drone.quit()
# time.sleep(100)

# print("drone.set_yaw(0.3):")

# drone.set_yaw(0.3)
# time.sleep(5)
# print("drone.set_yaw(-0.3):")
# drone.set_yaw(-0.3)

# time.sleep(5)
# print("drone.set_yaw(0):")
# drone.set_yaw(0)




