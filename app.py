import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import easyocr
from gtts import gTTS
from io import BytesIO
from googletrans import Translator

app = Flask(__name__)
CORS(app)

# Initialize OCR reader with improved settings
reader = easyocr.Reader(['en'], gpu=False)
translator = Translator()

def preprocess_image(image):
    """Enhance image for better OCR results"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return cv2.medianBlur(processed, 3)

@app.route('/')
def home():
    return send_from_directory('templates', 'index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        # Read and process image
        img_file = request.files['image']
        img_arr = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # Preprocess and OCR
        processed_img = preprocess_image(img)
        results = reader.readtext(processed_img, detail=1, paragraph=False)
        
        # Extract text and annotate image
        extracted_text = []
        annotated_img = img.copy()
        for detection in results:
            text = detection[1]
            confidence = detection[2]
            points = [tuple(map(int, point)) for point in detection[0]]
            
            extracted_text.append({'text': text, 'confidence': f"{confidence*100:.2f}%"})
            cv2.polylines(annotated_img, [np.array(points)], True, (0, 255, 0), 2)
            cv2.putText(annotated_img, text, points[0], 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Prepare response
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        full_text = ' '.join([t['text'] for t in extracted_text])
        
        # Generate audio
        tts = gTTS(text=full_text, lang='en')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'text': extracted_text,
            'image': img_base64,
            'audio': audio_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_lang = data.get('language', 'en')
        
        # Translate text
        translated = translator.translate(text, dest=target_lang)
        
        # Generate translated audio
        tts = gTTS(text=translated.text, lang=target_lang)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'translated_text': translated.text,
            'audio': audio_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)