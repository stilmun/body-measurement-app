from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from video_utils import process_video_and_measure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Ensure necessary folders exist
for folder in ["uploads", "annotated_frames", "static"]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        height = int(request.form['height'])
        file = request.files['video']

        if not file or file.filename == '':
            return "❌ No video uploaded", 400

        if not allowed_file(file.filename):
            return "❌ Unsupported file type. Please upload .mp4 or .mov", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"📦 Saved file to: {filepath}")
        results, annotated_path = process_video_and_measure(filepath, height)

        if "error" in results:
            print(f"🚫 Processing error: {results['error']}")
            return f"❌ {results['error']}", 500

        return render_template('results.html', measurements=results, image_path=annotated_path)

    except Exception as e:
        print(f"🔥 Unexpected error: {e}")
        return f"❌ An unexpected error occurred: {str(e)}", 500

if __name__ == '__main__':
    pass  # Run with gunicorn in production
