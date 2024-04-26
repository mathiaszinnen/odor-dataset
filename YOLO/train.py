from ultralytics import YOLO


model = YOLO("yolov8l.pt")
model.train(epochs=50, data='data/odor.yaml', name='yolov8l')
