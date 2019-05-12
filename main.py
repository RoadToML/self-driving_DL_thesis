from carla import Client, Transform, Rotation, Location

import carla
import random
import time

# starting proper work now

print('attempting to connect...') # DEBUG
client = Client('localhost', 2000, worker_threads = 12)
client.set_timeout(10.0)

client.load_world('/Game/Carla/Maps/single_left_bend')
print('connected!') # DEBUG

world = client.get_world()
# bp of all actors
blueprint_lib = world.get_blueprint_library()

# #################################################
# ADD SPEED LIMIT SIGN
# actor_list = world.get_actors()
# actor_transform = actor_list.filter('traffic.speed_limit.*')

# speed_sign_bp = blueprint_lib.find('static.prop.streetsign01')
# speed_sign_50_spawn = world.spawn_actor(speed_sign_bp, actor_transform[0])


# #################################################

vehicle_bp = blueprint_lib.find('vehicle.audi.tt')

# adding spawn point for car as per coordinates on unreal engine
spawn_points = world.get_map().get_spawn_points() 

print('spawning car') # DEBUG

vehicle_actor = world.spawn_actor(vehicle_bp, spawn_points[0])
vehicle_actor.set_autopilot(True)
time.sleep(2)
print(vehicle_actor.get_location())
print('DONE') # DEBUG

# #################################################
# Add Camera 
camera_bp = blueprint_lib.find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, spawn_points[0], attach_to = vehicle_actor)
camera.listen(lambda image: image.save_to_disk('output/%06d.png' %image.frame_number))

# #################################################

time.sleep(2)
vehicle_actor.set_simulate_physics(True)

location = vehicle_actor.get_location()
transform = vehicle_actor.get_transform()
print(transform)
print(location)

while True:
    print(vehicle_actor.get_velocity())
    time.sleep(1)

# my_map = world.get_map()

# waypoint_list = my_map.generate_waypoints(2.0)
# print(len(waypoint_list))

# waypoint = my_map.get_waypoint(vehicle_actor.get_location())
# print(waypoint)

# vehicle_actor.set_simulate_physics(False)
# c = 0
# for waypoint in waypoint_list:
#     if c == 20:
#         break
#     vehicle_actor.set_transform(waypoint.transform)
#     time.sleep(3)
#     print(vehicle_actor.get_location())
#     c += 1

# ##############################################################
# ##########this code allows for turning the car ###############
# ##############################################################
# while True:
#     transform = vehicle_actor.get_transform()
#     print(transform)
#     transform.rotation.yaw += 3.0
#     vehicle_actor.set_transform(transform)
#     time.sleep(0.5)
# ###############################################################

# location.y = 3.0
# actor.set_location(location)