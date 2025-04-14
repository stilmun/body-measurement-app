from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from video_utils import process_video_and_measure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Max 20MB upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        height = int(request.form['height'])
        file = request.files['video']

        if not file:
            return "No video uploaded", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ⬇️ Optional: resize the video to prevent memory overload
        import cv2
        cap = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_path = filepath.replace('.mp4', '_resized.mp4')
        out = cv2.VideoWriter(temp_path, fourcc, 10.0, (360, 640))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_small = cv2.resize(frame, (360, 640))
            out.write(frame_small)

        cap.release()
        out.release()
        filepath = temp_path

        # 👇 Call the main logic here
        results, annotated_path = process_video_and_measure(filepath, height)

        if "error" in results:
            return f"❌ {results['error']}", 500

        return render_template('results.html', measurements=results, image_path=annotated_path)

    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
