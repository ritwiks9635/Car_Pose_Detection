import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

MODEL_PATH = "car_view_fp16.tflite"
TEST_IMAGE_DIR = "test_images"
OUTPUT_CSV = "predictions.csv"

IMG_SIZE = 224

CLASS_NAMES = [
    "front",
    "frontleft",
    "frontright",
    "rear",
    "rearleft",
    "rearright",
    "unknown"
]

ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASS_NAMES)}


interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path):
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return None, None

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    pred_id = int(np.argmax(output))
    confidence = float(output[0][pred_id])

    return ID_TO_CLASS[pred_id], confidence

results = []

image_files = [
    f for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_files)} test images")

for img_name in tqdm(image_files, desc="Running predictions"):
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    label, conf = predict_image(img_path)

    if label is None:
        continue

    results.append({
        "image_name": img_name,
        "prediction": label,
        "confidence": round(conf, 4)
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nPrediction complete!")
print(f"Results saved to: {OUTPUT_CSV}")
print(df.head())
