import zmq

from babylog.logger import babylogger


class Publisher:
    def __init__(self, address: str, port: int, topic: str, context_io_threads: int = 1):
        self._topic = topic
        self._context = zmq.Context(context_io_threads)
        try:
            self._publisher = zmq.Socket(self._context, zmq.PUB)
            self._publisher.bind(f'tcp://{address}:{port}')
            babylogger.info(f'Publisher with topic ({topic}) succesfully binded to port {port}')
        except Exception as e:
            babylogger.error(f'could not initialize publisher: {e}')
            raise ValueError(f'could not initialize publisher: {e}')

    def send(self, data: bytes) -> bool:
        try:
            if self._publisher.send_string(self._topic, zmq.SNDMORE) is not None:
                babylogger.error(1)
                return False
            if self._publisher.send(data) is not None:
                babylogger.error(2)
                return False
            return True
        except Exception as e:
            babylogger.error(f'exception while sending with publisher: {e}')
            return False

    def shutdown(self):
        self._publisher.close()
