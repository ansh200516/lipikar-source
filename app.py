from flask import Flask, request, jsonify, send_file
import os
import threading
import uuid
import subprocess
from flask_cors import CORS
from threading import Lock
import PyPDF2  # Add this import for PDF page count

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150 MB limit, adjust as needed
CORS(app)  # Enable CORS to allow requests from your frontend

# Dictionary to store the status, progress, and result paths of OCR jobs
jobs = {}
# Lock to prevent concurrent access to test_pdf folder
lock = Lock()

TEST_PDF_FOLDER = './test_pdf'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Average processing time per unit (adjust based on actual performance)
AVERAGE_TIME_PER_UNIT = 5  # seconds per page/image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_files():
    # Create a unique inference ID
    inference_id = str(uuid.uuid4())

    # Get parameters
    language = request.form.get('language')
    output_type = request.form.get('output_type')
    file_type = request.form.get('file_type')

    if not language or not output_type or not file_type:
        return jsonify({'success': False, 'error': 'Required parameters not provided'}), 400

    # Get uploaded files
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    # Validate files
    for file in files:
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

    with lock:
        # Clear the test_pdf folder
        if os.path.exists(TEST_PDF_FOLDER):
            for filename in os.listdir(TEST_PDF_FOLDER):
                file_path = os.path.join(TEST_PDF_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(TEST_PDF_FOLDER)

        # Save files and determine the number of units
        number_of_units = 0
        if file_type == 'pdf':
            # Save the PDF file
            pdf_file = files[0]
            pdf_file_path = os.path.join(TEST_PDF_FOLDER, pdf_file.filename)
            pdf_file.save(pdf_file_path)
            # Get the number of pages in the PDF
            with open(pdf_file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                number_of_units = len(reader.pages)
        elif file_type == 'images':
            # Save image files
            for file in files:
                file.save(os.path.join(TEST_PDF_FOLDER, file.filename))
            number_of_units = len(files)
        else:
            return jsonify({'success': False, 'error': 'Unknown file type'}), 400

    # Calculate estimated time
    estimated_time = number_of_units * AVERAGE_TIME_PER_UNIT

    # Initialize job status
    jobs[inference_id] = {'status': 'processing', 'output_path': None, 'progress': 0}

    # Start OCR process in a separate thread
    threading.Thread(target=run_ocr, args=(inference_id, language, output_type)).start()

    return jsonify({'success': True, 'inference_id': inference_id, 'estimated_time': estimated_time}), 200

def run_ocr(inference_id, language, output_type):
    try:
        with lock:
            # Run the OCR script with the necessary arguments
            command = [
                'python', 'ocr.py',
                '--lang', language,
                '--input', TEST_PDF_FOLDER,
                '--output_type', output_type
            ]
            print(f"Running OCR command: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            # Read output line by line
            for line in process.stdout:
                print(f"OCR output: {line.strip()}")
                if line.startswith('PROGRESS:'):
                    # Update progress
                    try:
                        progress = int(line.strip().split(':')[1])
                        jobs[inference_id]['progress'] = progress
                    except ValueError:
                        pass  # Ignore invalid progress values
            process.wait()
            if process.returncode != 0:
                print(f"OCR processing failed for job {inference_id}")
                jobs[inference_id]['status'] = 'error'
                return
            if jobs[inference_id]['progress'] < 100:
                jobs[inference_id]['progress'] = 100
            # Determine the output file based on output_type
            output_file = None
            if output_type == 'pdf':
                output_dir = os.path.join('results', 'recognition_results_pdf')
                output_files = os.listdir(output_dir)
                if output_files:
                    # Assuming the output file is the first in the directory
                    output_file = os.path.join(output_dir, output_files[0])
            elif output_type == 'text':
                output_dir = os.path.join('results', 'recognition_results_txt')
                output_files = os.listdir(output_dir)
                if output_files:
                    output_file = os.path.join(output_dir, output_files[0])
            else:
                print(f"Invalid output type: {output_type}")
                jobs[inference_id]['status'] = 'error'
                return

            # Rename the output file to include the inference_id
            if output_file and os.path.exists(output_file):
                base, ext = os.path.splitext(output_file)
                new_output_file = f"{base}_{inference_id}{ext}"
                os.rename(output_file, new_output_file)
                jobs[inference_id]['status'] = 'done'
                jobs[inference_id]['output_path'] = new_output_file
                print(f"OCR completed successfully for job {inference_id}")
            else:
                print(f"Output file not found for job {inference_id}")
                jobs[inference_id]['status'] = 'error'
    except subprocess.CalledProcessError as e:
        print(f"OCR processing failed for job {inference_id}: {e}")
        print(f"Command stdout: {e.stdout}")
        print(f"Command stderr: {e.stderr}")
        jobs[inference_id]['status'] = 'error'
    except Exception as e:
        print(f"Unexpected error for job {inference_id}: {e}")
        import traceback
        traceback.print_exc()
        jobs[inference_id]['status'] = 'error'

@app.route('/status', methods=['GET'])
def check_status():
    inference_id = request.args.get('inference_id')
    if inference_id in jobs:
        status = jobs[inference_id]['status']
        progress = jobs[inference_id].get('progress', 0)
        response = {'success': True, 'status': status, 'progress': progress}
        if status == 'done':
            response['result'] = {'ocr_result_link': f'/result?inference_id={inference_id}'}
        return jsonify(response), 200
    else:
        return jsonify({'success': False, 'error': 'Invalid inference ID'}), 400

@app.route('/result', methods=['GET'])
def get_result():
    inference_id = request.args.get('inference_id')
    if inference_id in jobs and jobs[inference_id]['status'] == 'done':
        output_path = jobs[inference_id]['output_path']
        if os.path.exists(output_path):
            # Send the file as an attachment
            return send_file(output_path, as_attachment=True)
        else:
            return jsonify({'success': False, 'error': 'Output file not found'}), 404
    else:
        return jsonify({'success': False, 'error': 'Result not available or job not completed'}), 400

if __name__ == '__main__':
    # Run the app on all available IP addresses
    app.run(host='0.0.0.0', port=5000)
