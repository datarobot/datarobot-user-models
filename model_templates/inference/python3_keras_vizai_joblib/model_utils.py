# keras imports
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# scikit-learn imports
from sklearn.pipeline import Pipeline

# pandas/numpy imports
import pandas as pd
import numpy as np

import joblib
import io
import base64
import h5py
from PIL import Image
from pathlib import Path

# define constants

IMG_SIZE = 150
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


def get_imputation_img() -> str:
    """ black image in base64 str for data imputation filling """
    black_PIL_img = Image.fromarray(np.zeros(IMG_SHAPE, dtype="float32"), "RGB")
    return get_base64_str_from_PIL_img(black_PIL_img)


def get_img_obj_from_base64_str(b64_img_str: str) -> Image:
    """ given a base64 encoded image str get the PIL.Image object """
    b64_img = base64.b64decode(b64_img_str)
    b64_img = io.BytesIO(b64_img)
    return Image.open(b64_img)


def get_base64_str_from_PIL_img(pillowed_img: Image) -> str:
    """ given a PIL.Image object return base64 encoded str of the image object """
    buffer = io.BytesIO()
    pillowed_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue())


def img_preprocessing(pillowed_img: Image) -> np.ndarray:
    """ given a PIL.Image object resize, convert to RGB and return as np.array """
    img = pillowed_img.resize((IMG_SHAPE[:-1]), Image.LANCZOS)
    img = img.convert("RGB")
    img_arr = np.asarray(img, dtype="float32")
    img_arr = preprocess_input(img_arr)  # pixel scaling/color normalization
    return img_arr


def preprocessing_X_transform(data_df: pd.DataFrame, image_feature_name: str,) -> pd.DataFrame:
    """ Apply the preprocessing methods on the data before prediction for the model to work on """

    data_df = data_df.copy()
    if image_feature_name in data_df:
        data_df[image_feature_name] = data_df[image_feature_name].astype(bytes)
        data_df[image_feature_name] = data_df[image_feature_name].apply(get_img_obj_from_base64_str)
        data_df[image_feature_name] = data_df[image_feature_name].apply(img_preprocessing)
    return data_df


def pretrained_preprocess_input(img_arr: np.ndarray) -> np.ndarray:
    return preprocess_input(img_arr)


def reshape_numpy_array(data_series: pd.Series) -> np.ndarray:
    """ Convert pd.Series to numpy array and reshape it too """
    return np.asarray(data_series.to_list()).reshape(-1, *IMG_SHAPE)


def apply_image_data_preprocessing(x_data_df: pd.DataFrame, image_feature_name: str) -> np.ndarray:
    """ Image data preprocessing before fit """
    X_data_df = preprocessing_X_transform(x_data_df, image_feature_name)
    X_data = reshape_numpy_array(X_data_df[image_feature_name])
    return X_data


def convert_np_to_df(np_array, img_col) -> pd.DataFrame:
    """ simple utility to convert numpy array to dataframe """
    return pd.DataFrame(data=np_array, columns=[img_col])


def deserialize_estimator_pipeline(input_dir: str) -> Pipeline:
    """
    Load estimator pipeline from the given joblib file.

    Parameters
    ----------
    joblib_file_path: str
        The joblib file path to load from.

    Returns
    -------
    pipeline: Pipeline
        Constructed pipeline with necessary preprocessor steps and estimator to predict/score.
    """
    # load the dictionary obj from the joblib file
    joblib_file_path = Path(input_dir) / "artifact.joblib"
    estimator_dict = joblib.load(joblib_file_path)
    model = estimator_dict["model"]
    prep_pipeline = estimator_dict["preprocessor_pipeline"]
    with h5py.File(model, mode="r") as fp:
        keras_model = load_model(fp)

    pipeline = Pipeline([("preprocessor", prep_pipeline), ("estimator", keras_model)], verbose=True)
    return pipeline
