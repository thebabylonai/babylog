from tqdm import tqdm

from babylog.pubsub import Subscriber

sub = Subscriber(address="127.0.0.1", port=5555, topic="DEVICE_NAME")

for i in tqdm(range(10)):
    if sub.logged_data is not None:
        print(sub.logged_data.detection)
