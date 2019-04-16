from carla import Client, Transform, Location, Rotation

import random

print('attempting to connect...') # DEBUG
client = Client('localhost', 2000, worker_threads = 12)
client.set_timeout(10.0)
print('connected!') # DEBUG

world = client.get_world()

blueprint_library = world.get_blueprint_library()

# find specific blueprint
collision_sensor_bp = blueprint_library.find('sensor.other.collision')

# Choose a vehicle blueprint at random
vehicle_bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))
print(vehicle_bp) #DEBUG / INFORMATIVE

vehicles = blueprint_library.filter('vehicle.*')
bikes = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 2]
print(bikes) # DEBUG / INFORMATIVE

transform = Transform(Location(x = 230, y = 195, z = 40), Rotation(yaw = 180))

# try_spawn_actor can work too. without 'try' raises error. 
actor = world.spawn_actor(vehicle_bp, transform)
print(actor) # DEBUG / INFORMATIVE

# get list of all spawn points
spawn_points = world.get_map().get_spawn_points()

# Handling Actors
location = actor.get_location()
location.z += 10.0
actor.set_location(location)
print(actor.get_acceleration())
print(actor.get_velocity())

actor.simulate_physics(False)

actor.destroy()