import time


class TurnSession:
    def __init__(self, engine):
        self.engine = engine
        self.last_state = None

        # session lifecycle management
        now = time.time()
        self.created_ts = now
        self.last_active_ts = now

    def touch(self):
        """
        Called by server, used for session keep-alive / GC
        """
        self.last_active_ts = time.time()

    def feed_audio(self, audio):
        """
        audio: np.ndarray(float32)
        """
        # Consider active every time audio is received
        self.touch()

        result = self.engine.process(audio)
        self.last_state = result["state"]
        return result

        # # Report only when state changes
        # if result["state"] != self.last_state:
        #     self.last_state = result["state"]
        #     return result

        # return None
