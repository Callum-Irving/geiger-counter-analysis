import sys
import signal
import time
import csv
import logging
import os
from datetime import datetime

import RPi.GPIO as GPIO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Get data file from argv
assert len(sys.argv) == 2, "expected one command line argument"

GEIGER_PIN = 14
DATA_FILE = sys.argv[1]

# Seconds between refreshing counts
INTERVAL = 60

# Number of counts in the last INTERVAL
count = 0

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(GEIGER_PIN, GPIO.IN)


def exit_handler(_sig, _frame):
    """Cleanup GPIO on program exit."""
    logging.info("safely exiting")
    GPIO.cleanup()
    sys.exit(0)


def click_handler(ev):
    """Increment global count when Geiger tube is hit."""
    global count
    count += 1
    logging.debug("click")


# Register click and exit handlers
GPIO.add_event_detect(GEIGER_PIN, GPIO.FALLING, click_handler)
signal.signal(signal.SIGINT, exit_handler)

# Create data file if it doesn't exist already
if not os.path.isfile(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "CPM"])
    logging.info(f"created new file: {DATA_FILE}")
else:
    logging.info(f"{DATA_FILE} already exists, no need to create new file")

# Logging loop
while True:
    time.sleep(INTERVAL)
    cpm = count * (60 / INTERVAL)
    count = 0
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")

    logging.debug(f"CPM: {cpm}")
    with open(DATA_FILE, "a") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, cpm])
