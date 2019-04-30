from carla import Client

import carla
import random

# starting proper work now

print('attempting to connect...') # DEBUG
client = Client('localhost', 2000, worker_threads = 12)
client.set_timeout(10.0)

client.load_world('/Game/Carla/ExportedMaps/scenario_1_only_bends')
print('connected!') # DEBUG

world = client.get_world()
print(client.get_available_maps())