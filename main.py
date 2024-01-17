from ultralytics import YOLO
import numpy as np
import torch
import os
import shutil
from flask import Flask, render_template, request, send_from_directory
from datetime import datetime  # Import the datetime module


import uvicorn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


static_folder = os.path.join(os.getcwd(), 'static')

print(static_folder)


app = Flask(__name__)





def process_file(model_filename):
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    # print(file)

    if file.filename == '':
        return "render_template('index.html', message='No selected file')"

    if file:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(static_folder, 'uploads', filename)
        file.save(file_path)

        model = YOLO(model_filename)
        result = model.predict(file_path, save=True, show=False)

        result_path = result[0].save_dir

        if result_path:
            result_filename = os.path.basename(result_path)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            result_folder = f"{timestamp}_{result_filename}"
            dest_path = os.path.join(static_folder, result_folder)

            # if os.path.exists(dest_path):
            #     os.remove(dest_path)

            shutil.move(result_path, dest_path)
            print(dest_path)
            trimmed_path = dest_path.split('static', 1)[1]

            final_path = '/static' + trimmed_path + '/' + filename
            print('final_path = ', final_path)
            return final_path






@app.route('/')
def index():
    return render_template('index.html')



@app.route('/best_first', methods=['POST'])
def upload_best_first():
    return process_file('Models/best_first.pt')

@app.route('/best_second', methods=['POST'])
def upload_best_second():
    return process_file('Models/best_second.pt')

@app.route('/best_third', methods=['POST'])
def upload_best_third():
    return process_file('Models/best_third.pt')



if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(port=8000, host="0.0.0.0")