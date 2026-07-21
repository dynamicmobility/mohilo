import time
from rich.progress import track

for item in track(range(100), description="Downloading..."):
    time.sleep(0.05)