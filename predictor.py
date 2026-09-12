from ultralytics import YOLO 

model_name = "runs/classify/train/weights/best.pt"
test_file = "test_file.jpeg"

model = YOLO(model_name)

result = model(test_file)
data = result[0]
label = data.names[data.probs.top1] 
conf = data.probs.top1conf.item() 
print(f"{label} ({conf:.2%})")