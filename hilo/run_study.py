import time
import sched
import queue

class Trial:
    def __init__(self, cart, ipad, abort):
        self.sched = sched.scheduler(time.monotonic, abort.wait)
        self.cart, self.ipad, self.abort = cart, ipad, abort
        self.responses = queue.Queue()
        self.samples = []
        ipad.on_response = self._on_response

    def _repeat(self, period, fn, i=0):
        if time.monotonic() >= self.deadline:
            return
        fn()
        self.sched.enterabs(self.t0 + (i + 1) * period, 1, self._repeat,
                            (period, fn, i + 1))

    def _prompt(self):
        self.ipad.show_question(asked_at=time.monotonic() - self.t0)

    def _on_response(self, asked_at, value):
        self.responses.put((asked_at, time.monotonic() - self.t0, value))

    def run(self, duration=120.0):
        self.t0 = time.monotonic()
        self.deadline = self.t0 + duration
        self._repeat(1.0, lambda: self.samples.append(
            (time.monotonic() - self.t0, self.cart.read())))
        self._repeat(30.0, self._prompt)
        self.sched.run()
        return self.samples, list(self.responses.queue)

def main():
    pass
    # Create gp, objective, and acquisition function
    
    # Connect to iPad. Run any diagnostics/check connection
    
    for trial in range(NUM_TRIALS):
        # fit the GP to the current data 
        
        # argmax the acqusition function
        
        # send the action to the exo device
        
        # wait 2 minutes until metabolics are ready. every 30 seconds, make a comfort level query to the user
        pass

if __name__ == 'main':
    main()