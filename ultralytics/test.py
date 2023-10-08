from ultralytics import YOLO


model = YOLO("runs/detect/yolov8l/weights/last.pt")

model.val(data='data/odor.yaml', save_json=True, name='eval_yolov8l')
