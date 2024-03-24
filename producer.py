from confluent_kafka import Producer

class MyProducer:
    def __init__(self, bootstrap_servers, topic): 
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": self.bootstrap_servers, "group.id": "my"})

    def delivery_report(self, err, msg):
        if err is not None:
            print("Ошибка при отправке сообщения: {}".format(err))
        else:
            print("Сообщение успешно отправлено в топик {}".format(msg.topic()))

    def produce_message(self, message_key, message_value):
        self.producer.produce(
            self.topic,
            key=str(message_key),
            value=str(message_value),
            callback=self.delivery_report)
        
    def flush(self):
        self.producer.flush()