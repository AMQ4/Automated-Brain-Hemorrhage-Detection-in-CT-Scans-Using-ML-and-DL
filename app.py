from flask import Flask, request, render_template, redirect, url_for, flash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'dcm'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File uploaded successfully')
        return redirect(url_for('detect'))
    else:
        flash('Invalid file type. Please upload a valid CT scan image (.jpg, .jpeg, .png, .dcm).')
        return redirect(url_for('index'))

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Simulate detection logic. This should be replaced with actual ML model inference."""
    if request.method == 'POST':
        # Here, you would process the uploaded file and run the model
        result = "No Hemorrhage Detected"  # Placeholder result
        flash(f'Result: {result}')
        return redirect(url_for('index'))

    return render_template('detect.html')

if __name__ == '__main__':
    app.run(debug=True)
