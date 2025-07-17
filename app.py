import os
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

async def save_temp_file(file: UploadFile):
    temp_filename = f"/tmp/{uuid.uuid4()}.png"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())
    return temp_filename

def preprocess(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((220, 155))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dim
    image = np.expand_dims(image, axis=0)   # Add batch dim
    print(f"Preprocessed image shape: {image.shape}, dtype: {image.dtype}")
    return image

@profile
def predict_similarity(image1_path, image2_path):
    img1 = preprocess(image1_path)
    img2 = preprocess(image2_path)
    prediction = model.predict([img1, img2])
    print(f"Raw model prediction output: {prediction}")
    return float(prediction[0][0])

@app.post("/signature-verify/")
async def signature_verify(signature_image: UploadFile = File(...), database_image: UploadFile = File(...)):
    try:
        temp_sig = await save_temp_file(signature_image)
        temp_db = await save_temp_file(database_image)
        score = predict_similarity(temp_sig, temp_db)

        if not np.isfinite(score):
            raise ValueError(f"Invalid prediction score: {score}")

        verdict = "match" if score >= 0.9 else ("similar" if score >= 0.7 else "no_match")
        return JSONResponse({
            "score": round(float(score) * 100, 2),
            "result": verdict
        })
    except Exception as e:
        return JSONResponse({
            "score": None,
            "result": "error",
            "detail": str(e)
        }, status_code=500)
