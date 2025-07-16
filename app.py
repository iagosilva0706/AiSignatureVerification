import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import uuid
import gdown


def download_model():
    url = 'https://drive.google.com/file/d/1O6F8-dcxAe2jwjrBV6jZLqmKFw8WtaHl/view?usp=drive_link'  # Replace with actual Google Drive file ID
    local_filename = 'model.h5'

    if not os.path.exists(local_filename):
        print("Downloading model from Google Drive...")
        gdown.download(url, local_filename, quiet=False)
        print(f"Model downloaded. File size: {os.path.getsize(local_filename)} bytes.")

    return local_filename


# Download and load model once
model_path = download_model()
model = tf.keras.models.load_model(model_path)

# Initialize FastAPI
app = FastAPI()


@app.post("/signature-verify/")
def signature_verify(signature_image: UploadFile = File(...), database_image: UploadFile = File(...)):
    temp_sig = save_temp_file(signature_image)
    temp_db = save_temp_file(database_image)

    score = predict_similarity(temp_sig, temp_db)

    verdict = "match" if score >= 0.9 else ("similar" if score >= 0.7 else "no_match")

    return JSONResponse({
        "score": round(float(score) * 100, 2),
        "result": verdict
    })


def save_temp_file(file: UploadFile):
    temp_filename = f"/tmp/{uuid.uuid4()}.png"
    with open(temp_filename, "wb") as f:
        f.write(file.file.read())
    return temp_filename


def preprocess(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((220, 155))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image


def predict_similarity(image1_path, image2_path):
    img1 = preprocess(image1_path)
    img2 = preprocess(image2_path)
    prediction = model.predict([img1, img2])
    return float(prediction[0][0])
