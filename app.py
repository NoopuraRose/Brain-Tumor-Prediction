from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from predictor import predict_image

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("mri_image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please choose an MRI image to upload."
        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a PNG, JPG, JPEG, or WEBP image."
        else:
            filename = secure_filename(uploaded_file.filename)
            saved_filename = f"{uuid4().hex}_{filename}"
            image_path = UPLOAD_FOLDER / saved_filename
            uploaded_file.save(image_path)

            try:
                label, confidence = predict_image(image_path)
                image_url = f"/static/uploads/{saved_filename}"
            except Exception:
                image_path.unlink(missing_ok=True)
                error = "The image could not be processed. Please upload a valid MRI image."

    return render_template(
        "index.html",
        label=label,
        confidence=confidence,
        image_url=image_url,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
