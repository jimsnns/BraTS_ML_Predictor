import base64
from functools import lru_cache
from io import BytesIO

import nibabel as nib
import numpy as np
import segmentation_models_3D as sm
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

app = FastAPI(title="BraTS ML Predictor API", version="1.0.0")

CLASS_NAMES = [
    "Not Tumor",
    "Non-Enhancing Tumor class 1",
    "Edema class 2",
    "Enhancing Tumor class 3",
]


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


@lru_cache(maxsize=1)
def get_model():
    wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    f_score = sm.metrics.FScore()
    recall = tf.keras.metrics.Recall()
    iou = sm.metrics.IOUScore()

    return load_model(
        "training_3d_200epoch_final_default.hdf5",
        custom_objects={
            "iou_score": iou,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "f1-score": f_score,
            "precision": precision,
            "recall": recall,
        },
    )


def _load_nifti_from_upload(uploaded_file: UploadFile) -> np.ndarray:
    file_content = uploaded_file.file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail=f"{uploaded_file.filename} is empty.")
    file_io = BytesIO(file_content)
    fileholder = nib.FileHolder(fileobj=file_io)
    img = nib.Nifti1Image.from_file_map({"header": fileholder, "image": fileholder})
    return img.get_fdata()


def _load_and_scale(uploaded_file: UploadFile, scaler: MinMaxScaler) -> np.ndarray:
    img_data = _load_nifti_from_upload(uploaded_file)
    return scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)


def _prepare_volume(t2: np.ndarray, t1ce: np.ndarray, flair: np.ndarray) -> np.ndarray:
    combined_data = np.stack([t2, t1ce, flair], axis=3)
    combined_data = combined_data[56:184, 56:184, 13:141]
    return np.expand_dims(combined_data, axis=0)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    t2: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    for uploaded in (t2, t1ce, flair):
        if not uploaded.filename.endswith(".nii"):
            raise HTTPException(status_code=400, detail=f"{uploaded.filename} must be a .nii file.")

    scaler = MinMaxScaler()
    t2_data = _load_and_scale(t2, scaler)
    t1ce_data = _load_and_scale(t1ce, scaler)
    flair_data = _load_and_scale(flair, scaler)

    model_input = _prepare_volume(t2_data, t1ce_data, flair_data)
    model = get_model()

    prediction = model.predict(model_input)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :].astype(np.uint8)

    buffer = BytesIO()
    np.save(buffer, prediction_argmax)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "classes": CLASS_NAMES,
        "prediction_shape": prediction_argmax.shape,
        "prediction_npy_base64": encoded,
    }
