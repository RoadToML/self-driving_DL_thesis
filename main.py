from carla import Client, Transform, Location, Rotation

import carla
import random

print('attempting to connect...') # DEBUG
client = Client('localhost', 2000, worker_threads = 12)
client.set_timeout(10.0)
print('connected!') # DEBUG

client.load_world('/Game/Carla/Maps/Town01')
world = client.get_world()

blueprint_library = world.get_blueprint_library()

# Create the camera BP
camera_bp = blueprint_library.find('sensor.camera.rgb')

# init setting for camera
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera_bp.set_attribute('fov', '110')
# time between sensor captures
camera_bp.set_attribute('sensor_tick', '1.0')

# location of camera 
camera_transform = Transform(Location(x=0.8, z=1.7))

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

camera = world.spawn_actor(camera_bp, camera_transform, attach_to=actor)
camera.listen(lambda image: image.save_to_disk('output/%06d.png'% image.frame_number))

# Handling Actors
location = actor.get_location()
location.z += 10.0
actor.set_location(location)
print(actor.get_acceleration())
print(actor.get_velocity())

#actor.set_simulate_physics(False)

#actor.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
actor.set_autopilot(True)

#actor.destroy()
