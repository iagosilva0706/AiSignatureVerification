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

print("Model input shape:", model.input_shape)  # Debug

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
    image = image.resize((155, 220))  # width=155, height=220 as expected by model
    image_np = np.array(image)        # shape (220, 155)
    image_np = image_np / 255.0       # normalize
    image_np = np.expand_dims(image_np, axis=-1)  # add channel dimension -> (220, 155, 1)
    image_np = np.expand_dims(image_np, axis=0)   # add batch dimension -> (1, 220, 155, 1)
    print("Preprocessed image shape:", image_np.shape)  # Debug
    return image_np

@profile
def predict_similarity(image1_path, image2_path):  # Profiled function
    img1 = preprocess(image1_path)
    img2 = preprocess(image2_path)
    prediction = model.predict([img1, img2])
    return float(prediction[0][0])
