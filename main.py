from carla.client import make_carla_client, CarlaClient
from carla.settings import CarlaSettings
from carla.sensor import Camera, Lidar
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

try:
    print('attempting connection...')
    with make_carla_client('localhost', 2000, timeout = 15) as client:

        print('connected')
        print(dir(client))
        t = client.read_data()
        print(t)



    client = client.disconnect()
    print('disconnected')

except KeyboardInterrupt:
    print('good bye!')