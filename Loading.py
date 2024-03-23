import itertools
import threading
import time
import sys

class Loading:
    def __init__(self):
        self.done = False

    def animate(self):
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if self.done:
                break
            # sys.stdout.write('\rgenerating response ' + c)
            # sys.stdout.flush()
            print("generating synopsis ", c, end="\r", flush=True)
            time.sleep(0.1)

    def __call__(self, done):
        t = threading.Thread(target=self.animate)
        t.start()
        self.done = done
