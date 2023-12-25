from flask import Flask, render_template, request, jsonify
import base64
import os
import numpy as np
import imutils
import cv2
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours

app = Flask(__name__)

# Load the pre-trained model
network = load_model('network')  # Adjust the path

img = cv2.imread(image_data)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
invertion = 255 - adaptive
def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]
    return conts

conts = find_contours(invertion.copy())

def extract_roi(img, x, y, w, h):
    roi = img[y:y+h, x:x+w]
    return roi

def thresholding(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def resize_img(img, size):
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized

def normalization(img):
    img = img.astype('float32') / 255.0  # convert to floating point
    img = np.expand_dims(img, axis=-1)  # add depth
    return img

def process_box(gray, x, y, w, h):
    roi = extract_roi(gray, x, y, w, h)
    thresh = thresholding(roi)
    resized = resize_img(thresh, (28, 28))
    normalized = normalization(resized)
    cv2_imshow(resized)  # Show result
    return (normalized, (x, y, w, h))

# Fungsi untuk melakukan OCR pada gambar
def do_ocr(image_data):
    # Lokasi output teks dari OCR
    output_path = 'C:\\Users\\nayla\\Desktop\\ml\\'

    # Simpan data gambar ke file
    image_path = os.path.join(output_path, 'captured_image.png')
    with open(image_path, 'wb') as img_file:
        img_file.write(base64.b64decode(image_data))

    # Load the captured image for further processing
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    invertion = 255 - adaptive

    # Image processing code (similar to the previous response)
    conts = find_contours(invertion.copy())
    detected_char = []

    for c in conts:
        (x, y, w, h) = cv2.boundingRect(c)

        min_w, max_w = 10, 160
        min_h, max_h = 14, 140

        if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
            detected_char.append(process_box(gray, x, y, w, h))

    pixels = np.array([px[0] for px in detected_char], dtype='float32')
    boxes = [box[1] for box in detected_char]

    digits = '0123456789'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWZYZ'
    char_list = digits + letters
    char_list = [ch for ch in char_list]

    preds = network.predict(pixels)
    result_text = ""

    # Char Prediction
    for p in preds:
        char_idx = np.argmax(p)
        result_text += char_list[char_idx]

    # Hapus file output teks setelah digunakan
    os.remove(image_path)

    return result_text

# Route untuk tampilan utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menerima data gambar dari front-end
@app.route('/capture', methods=['POST'])
def capture():
    image_data = request.form['imageData']
    result = do_ocr(image_data)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
