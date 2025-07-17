import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import uuid
import gdown
from keras.saving import register_keras_serializable
from memory_profiler import profile  # Added for memory profiling

# Register custom function
@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def download_model():
    url = 'https://drive.google.com/uc?id=1O6F8-dcxAe2jwjrBV6jZLqmKFw8WtaHl'
    local_filename = 'model.h5'

    if not os.path.exists(local_filename):
        print("Downloading model from Google Drive...")
        gdown.download(url, local_filename, quiet=False)
        file_size = os.path.getsize(local_filename)
        print(f"Model downloaded. File size: {file_size} bytes.")

        if file_size < 10000:
            raise RuntimeError(f"Downloaded model file too small: {file_size} bytes. Download likely failed.")

    return local_filename

model_path = download_model()
model = tf.keras.models.load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance})

app = FastAPI()

def save_temp_file(file: UploadFile):
    temp_filename = f"/tmp/{uuid.uuid4()}.png"
    with open(temp_filename, "wb") as f:
        f.write(file.file.read())
    return temp_filename

def preprocess(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((220, 155))  # width=220, height=155 as per model input
    image = np.array(image) / 255.0
    # Expand dims to (1, 220, 155, 1)
    image = np.expand_dims(image, axis=-1)  # add channel dim -> (220, 155, 1)
    image = np.expand_dims(image, axis=0)   # add batch dim -> (1, 220, 155, 1)
    return image.astype(np.float32)

@profile
def predict_similarity(image1_path, image2_path):
    img1 = preprocess(image1_path)
    img2 = preprocess(image2_path)
    prediction = model.predict([img1, img2])
    score = float(prediction[0][0])
    if not math.isfinite(score):
        score = 0.0  # fallback if invalid
    return score

@app.post("/signature-verify/")
def signature_verify(signature_image: UploadFile = File(...), database_image: UploadFile = File(...)):
    temp_sig = save_temp_file(signature_image)
    temp_db = save_temp_file(database_image)
    try:
        score = predict_similarity(temp_sig, temp_db)
        if not math.isfinite(score):
            raise ValueError(f"Model returned non-finite score: {score}")
        verdict = "match" if score >= 0.9 else ("similar" if score >= 0.7 else "no_match")
        return JSONResponse({
            "score": round(score * 100, 2),
            "result": verdict
        })
    except Exception as e:
        return JSONResponse({
            "score": None,
            "result": "error",
            "detail": str(e)
        })
