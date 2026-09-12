from ultralytics import YOLO

model_name = "yolov8n-cls.pt"
dataset = "Dataset/"
test_file = "test_file.jpeg"

model = YOLO(model_name)

model.train(data = dataset, epochs = 10, imgsz = 224)

result = model(test_file)
data = result[0]
label = data.names[data.probs.top1] 
conf = data.probs.top1conf.item() 
print(f"{label} ({conf:.2%})")