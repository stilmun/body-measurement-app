from flask import Flask, render_template, request, redirect, url_for
from video_utils import process_video_and_measure
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

if not os.path.exists("uploads"):
    os.makedirs("uploads")

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        video = request.files["video"]
        height_cm = float(request.form["height_cm"])

        video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        video.save(video_path)

        measurements, annotated_img_path = process_video_and_measure(video_path, height_cm)

        return render_template("results.html", measurements=measurements, image_path=annotated_img_path)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
