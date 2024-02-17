import cv2 as cv
import os
import glob

class VehicleDetector:
    def __init__(self, cfg_path, weights_path):
        # Load the YOLOv4 model
        self.net = cv.dnn_DetectionModel(cfg_path, weights_path)
        self.net.setInputSize(224, 224)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        self.classes_of_interest = [2, 3, 4, 6]
        
    def predict_vehicle(self, image):
        try:
            # Resize the image to the input size expected by the model
            #frame = cv.resize(image, (224, 224))
            classes, _, _ = self.net.detect(image, confThreshold=0.1, nmsThreshold=0.2)
            return "occupied" if any(class_id in classes for class_id in self.classes_of_interest) else "unoccupied"
        except Exception as e:
            print(f"Error predicting occupancy for {filepath}: {str(e)}")
            return "Error during prediction"
    
    

# Example usage:
#detector = VehicleDetector('yolov4.cfg', 'yolov4.weights')

# Now detector is always ready, and you can call predict_vehicle or scan_directory_and_predict anytime
# For example, to predict a single image:
# result = detector.predict_vehicle('path_to_image.jpg')
# print(result)


