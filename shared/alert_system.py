import cv2
import os
from datetime import datetime

ALERT_FOLDER = "alerts"

def trigger_violence_alert(frame):

    if not os.path.exists(ALERT_FOLDER):
        os.makedirs(ALERT_FOLDER)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{ALERT_FOLDER}/violence_{timestamp}.jpg"

    cv2.imwrite(filename, frame)

    return filename