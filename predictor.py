from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "classify" / "train" / "weights" / "best.pt"
model = YOLO(str(MODEL_PATH))


def predict_image(image_path: str | Path) -> tuple[str, float]:
	"""Classify an MRI image and return its label and confidence."""
	result = model(str(image_path))[0]
	label = result.names[result.probs.top1]
	confidence = result.probs.top1conf.item()
	return label, confidence


if __name__ == "__main__":
	label, confidence = predict_image(BASE_DIR / "test_file.jpeg")
	print(f"{label} ({confidence:.2%})")