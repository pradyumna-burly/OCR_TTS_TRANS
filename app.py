from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import easyocr
import cv2
import numpy as np
import base64
import pyttsx3  # Added for TTS functionality

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

reader = easyocr.Reader(['en'], gpu=False)

# Initialize TTS engine
tts_engine = pyttsx3.init()

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image_data = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Resize image for OCR
    resized_img = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)

    result = reader.readtext(resized_img)
    annotated_img = resized_img.copy()
    extracted_text = []

    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        confidence = detection[2]
        extracted_text.append({'text': text, 'confidence': f"{confidence * 100:.2f}"})
        cv2.rectangle(annotated_img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(annotated_img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Combine extracted text into a single string
    text_to_read = '\n'.join([item['text'] for item in extracted_text])

    # Perform Text-to-Speech (TTS)
    audio_path = None
    if text_to_read:
        audio_path = 'static/audio/output.mp3'
        tts_engine.save_to_file(text_to_read, audio_path)
        tts_engine.runAndWait()

    return jsonify({
        'text': extracted_text,
        'image': img_base64,
        'audio': f'/{audio_path}' if audio_path else None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
