
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import yaml

# class YOLODetector:
#     def __init__(self, model_path="models/yolo11/best.pt", config_path="config/yolo_config.yaml"):
#         """Khởi tạo YOLO detector"""
#         self.model = YOLO(model_path)
        
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         self.class_names = self.config['classes']['names']
    
#     def detect(self, image, conf_threshold=0.5, iou_threshold=0.45):
#         """
#         Phát hiện biển báo trong ảnh
        
#         Args:
#             image: numpy array hoặc đường dẫn đến ảnh
#             conf_threshold: ngưỡng confidence
#             iou_threshold: ngưỡng IOU cho NMS
        
#         Returns:
#             results: dictionary chứa thông tin detection
#         """
#         if isinstance(image, str):
#             image = cv2.imread(image)
        
#         # Chạy inference
#         results = self.model.predict(
#             image,
#             conf=conf_threshold,
#             iou=iou_threshold,
#             verbose=False
#         )
        
#         detections = []
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])
                
#                 detections.append({
#                     'bbox': [int(x1), int(y1), int(x2), int(y2)],
#                     'confidence': conf,
#                     'class_id': cls,
#                     'class_name': self.class_names.get(cls, f"class_{cls}")
#                 })
        
#         return {
#             'detections': detections,
#             'num_detections': len(detections),
#             'image_shape': image.shape
#         }
    
#     def visualize(self, image, detections, save_path=None):
#         """Vẽ bounding boxes lên ảnh"""
#         if isinstance(image, str):
#             image = cv2.imread(image)
        
#         image_copy = image.copy()
        
#         for det in detections['detections']:
#             x1, y1, x2, y2 = det['bbox']
#             conf = det['confidence']
#             label = det['class_name']
            
#             # Vẽ bounding box
#             cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#             # Vẽ label
#             label_text = f"{label}: {conf:.2f}"
#             cv2.putText(image_copy, label_text, (x1, y1-10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         if save_path:
#             cv2.imwrite(save_path, image_copy)
        
#         return image_copy

# if __name__ == "__main__":
#     detector = YOLODetector()
#     # Test detector
#     # results = detector.detect("test_image.jpg")
#     # print(results)

import cv2
import numpy as np
from ultralytics import YOLO
import yaml


class YOLODetector:
    def __init__(self, model_path="models/yolo11/train/weights/best.pt", config_path="config/yolo_config.yaml"):
        """Khởi tạo YOLO detector"""
        self.model = YOLO(model_path)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Fix lỗi key YAML
        self.class_names = {int(k): v for k, v in self.config['classes']['names'].items()}

    def detect(self, image, conf_threshold=0.5, iou_threshold=0.45):

        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Không đọc được ảnh. Kiểm tra đường dẫn hoặc định dạng.")

        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.class_names.get(cls, f"class_{cls}")
                })

        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': image.shape
        }

    def visualize(self, image, detections, save_path=None):

        if isinstance(image, str):
            image = cv2.imread(image)

        image_copy = image.copy()

        for det in detections['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = det['class_name']

            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{label}: {conf:.2f}"
            cv2.putText(image_copy, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, image_copy)

        return image_copy


if __name__ == "__main__":
    detector = YOLODetector()
