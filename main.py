from carla import Client, Transform, Rotation, Location

import carla
import random
import time

# starting proper work now

print('attempting to connect...') # DEBUG
client = Client('localhost', 2000, worker_threads = 12)
client.set_timeout(10.0)

client.load_world('/Game/Carla/ExportedMaps/only_bends_v2')
print('connected!') # DEBUG

world = client.get_world()

blueprint_lib = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_lib.filter('vehicle.audi.*'))

spawn_points = world.get_map().get_spawn_points()
point = spawn_points[0]
print(point)


print('spawning car') # DEBUG

# adding spawn point for car as per coordinates on unreal engine
car_spawn_point = Transform(Location(x=-391.470120, y=-2.151211, z=7.446297))

vehicle_actor = world.spawn_actor(vehicle_bp, point)
print('DONE') # DEBUG
