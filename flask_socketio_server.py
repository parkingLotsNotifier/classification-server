import base64
import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
from vehicle_detector import (
    VehicleDetector,
)  # Assuming you have a VehicleDetector class as defined earlier
from datetime import datetime

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

detector = VehicleDetector("yolov4.cfg", "yolov4.weights")  # Load your model


@socketio.on("image_list")
def receive_image_list(image_list):
    predictions = []
    for index, base64_image in enumerate(image_list):
        # Decode each base64-encoded image data
        image_data = base64.b64decode(base64_image.split(",")[0])
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Process each image here and generate a prediction
        prediction = detector.predict_vehicle(img)
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Predicted: {prediction} for image index: {index}"
        )
        predictions.append(prediction)

    # Send the list of predictions back to the client
    emit("predictions", predictions)
    print("batch emitted back to CPPS server")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8001, host="192.168.1.212")
