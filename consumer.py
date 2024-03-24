from confluent_kafka import Consumer
import json


class MyConsumer:
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = Consumer({"bootstrap.servers": self.bootstrap_servers, "group.id": "my"})
        self.consumer.subscribe([topic])

    def poll(self):
        while True:
            msg = self.consumer.poll(100)
            if msg is not None and msg.value() is not None:
                return json.loads(msg.value().decode('utf-8'))
            