from ultralytics import YOLO


model = YOLO("yolov5l.pt")
model.train(epochs=50, data='data/odor.yaml', name='yolov5l')
