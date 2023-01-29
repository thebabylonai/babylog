import threading
import queue

import zmq

from babylog.logger import babylogger
from babylog.deserialize import LoggedPrediction


class Publisher:
    def __init__(
        self, address: str, port: int, topic: str, context_io_threads: int = 1
    ):
        self._topic = topic
        self._context = zmq.Context(context_io_threads)
        try:
            self._publisher = zmq.Socket(self._context, zmq.PUB)
            self._publisher.bind(f"tcp://{address}:{port}")
            babylogger.info(
                f"Publisher with topic ({topic}) succesfully binded to port {port}"
            )
        except Exception as e:
            babylogger.error(f"could not initialize publisher: {e}")
            raise ValueError(f"could not initialize publisher: {e}")

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
            babylogger.error(f"exception while sending with publisher: {e}")
            return False

    def shutdown(self):
        self._publisher.close()


class Subscriber:
    def __init__(
        self,
        address: str,
        port: int,
        topic: str,
        max_data_history: int = 10,
        context_io_threads: int = 1,
    ):
        self._topic = topic
        self._max_data_history = max_data_history
        self._shutdown = False

        self._context = zmq.Context(context_io_threads)
        self._subscriber = zmq.Socket(self._context, zmq.SUB)
        self._subscriber.setsockopt(zmq.RCVTIMEO, 1000)

        try:
            self._subscriber.connect(f"tcp://{address}:{port}")
            self._subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
            babylogger.info(
                f"subscriber with topic ({topic}) successfully connected to port {port}"
            )
        except Exception as e:
            babylogger.error(f"could not setup subscriber: {e}")
            raise
        self._mutex = threading.Lock()
        self._data = b""
        self._data_queue = queue.Queue(maxsize=1000)
        threading.Thread(target=self.receive).start()

    def shutdown(self):
        babylogger.info(f"shutting down {self._topic} stream.")
        self._shutdown = True

    def receive(self):
        while not self._shutdown:
            try:
                if self._subscriber.poll(1000, zmq.POLLIN):
                    recv_msgs = self._subscriber.recv_multipart()
                    assert len(recv_msgs) == 2
                    self._mutex.acquire()
                    self._data = recv_msgs[1]
                    self._data_queue.put(self._data)
                    self._mutex.release()
                else:
                    pass
            except Exception as e:
                babylogger.error(f"could not receive message: {e}")
        babylogger.info(f"{self._topic} stream shut down")

    @property
    def data(self):
        self._mutex.acquire()
        data = self._data
        self._mutex.release()

        return data

    @property
    def logged_data(self, max_timeout=1):
        try:
            data = self._data_queue.get(timeout=max_timeout)
        except Exception as e:
            babylogger.info(f"could not get message: {e}")
            return None
        return LoggedPrediction.from_bytes(data)
