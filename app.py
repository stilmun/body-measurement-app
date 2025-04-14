from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from video_utils import process_video_and_measure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB

# Ensure necessary folders exist
for folder in ["uploads", "annotated_frames", "static"]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        height = int(request.form['height'])
        file = request.files['video']

        if not file:
            return "❌ No video uploaded", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Only ffmpeg-based preprocessing is used (handled inside video_utils)
        results, annotated_path = process_video_and_measure(filepath, height)

        if "error" in results:
            return f"❌ {results['error']}", 500

        return render_template('results.html', measurements=results)

    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}", 500

if __name__ == '__main__':
    pass  # Gunicorn will run this on Render
