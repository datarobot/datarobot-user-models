"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import base64
from pathlib import Path
from typing import Tuple
import io

import h5py
import joblib
from PIL import Image

# keras imports
import tensorflow
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    Add,
    GlobalAveragePooling2D,
    Input,
)  # core layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV3Small

# scikit-learn imports
from sklearn.preprocessing import LabelBinarizer, label_binarize, FunctionTransformer
from sklearn.pipeline import Pipeline

# pandas/numpy imports
import pandas as pd
import numpy as np

from typing import List, Iterator

# define constants
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.33
SEED = 4321
NUM_CLASSES = 2


def get_img_obj_from_base64_str(b64_img_str: str) -> Image:
    """given a base64 encoded image str get the PIL.Image object"""
    b64_img = base64.b64decode(b64_img_str)
    b64_img = io.BytesIO(b64_img)
    return Image.open(b64_img)


def img_preprocessing(pillowed_img: Image) -> np.ndarray:
    """given a PIL.Image object resize, convert to RGB and return as np.array"""
    img = pillowed_img.resize((IMG_SHAPE[:-1]), Image.LANCZOS)
    img = img.convert("RGB")
    img_arr = np.asarray(img, dtype="float32")
    return img_arr


def extract_features_from_pretrained_network(
    generator: Iterator[tuple], sample_count: int, base_model: Model
) -> Tuple[np.ndarray, np.ndarray]:
    """extract the features using the CNN base model"""
    conv_base_actvn_output_shape = tuple(base_model.layers[-1].output.shape[1:])
    features = np.zeros((sample_count, *conv_base_actvn_output_shape))
    labels = np.zeros((sample_count))
    i = 0

    # using base model to predict and extract the features
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch.squeeze()
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break

    return features, labels


def get_all_callbacks() -> List[tensorflow.keras.callbacks.Callback]:
    """List of all keras callbacks"""
    es = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="auto", min_delta=1e-3)
    return [es]


def get_image_augmentation_gen(X_data, y_data, bs, seed) -> Iterator[tuple]:
    """Generator which yields tuple of image data and corresponding labels in np.array"""

    # Note DataRobot currently has its own image augmentation functionality that will be applied
    # We are using the ImageDataGenerator for convenience only
    datagen = ImageDataGenerator()

    datagen.fit(X_data)

    image_aug_generator = datagen.flow(x=X_data, y=y_data, batch_size=bs, seed=seed)
    return image_aug_generator


def get_pretrained_base_model() -> Model:
    """A base pretrained model to build on top of"""
    weights_file = "mobilenetv3_small.h5"
    pretrained_model = MobileNetV3Small(
        include_top=False, input_shape=IMG_SHAPE, weights=weights_file
    )
    pretrained_model.trainable = False
    pool = GlobalAveragePooling2D()(pretrained_model.layers[-1].output)
    return Model(inputs=pretrained_model.inputs, outputs=pool)


def create_image_binary_classification_model(input_shape) -> Model:
    """Creates a Keras image classification model that will use the pre-trained base model above as a featurizer,
    i.e. we will feed the data through the pre-trained MobileNetV3 and extract the last layer of features,
    which this Keras model will then use to make binary predictions
    """
    inputs = Input(shape=input_shape)

    # Main Branch
    X = Dense(64, activation="relu")(inputs)
    X = BatchNormalization()(X)
    X = Dropout(0.05)(X)
    X = Dense(1, activation="sigmoid")(X)
    X = BatchNormalization()(X)

    # Skip Connection
    skip = Dense(1)(inputs)  # linear
    skip = BatchNormalization()(skip)

    outputs = Add()([X, skip])
    outputs = Dense(1, activation="sigmoid")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="keras_vizai")

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


def get_transformed_train_validation_split(
    X_df: pd.DataFrame, y_series: pd.Series, class_order: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """split train/test data after apply label encoder on y"""
    assert len(X_df) == len(y_series)
    # preprocessing steps
    y_series = label_binarize(y_series, classes=class_order)

    # convert np.ndarray to pd.Series
    y_series = pd.Series(y_series.squeeze())

    # split train/test data
    msk = np.random.rand(len(X_df)) < TEST_SIZE
    x_test_data = X_df[msk]
    x_train_data = X_df[~msk]

    y_test_data = y_series[msk]
    y_train_data = y_series[~msk]

    return x_train_data, x_test_data, y_train_data, y_test_data


def preprocessing_X_transform(data_df: pd.DataFrame, image_feature_name: str) -> pd.DataFrame:
    """Because DataRobot stores images as base64 strings, we need to transform them into pixel values by
    first transforming the strings to bytes, then transforming them into an image, and finally into an
    numpy array of pixels
    """

    data_df = data_df.copy()
    data_df[image_feature_name] = data_df[image_feature_name].astype(bytes)
    data_df[image_feature_name] = data_df[image_feature_name].apply(get_img_obj_from_base64_str)
    data_df[image_feature_name] = data_df[image_feature_name].apply(img_preprocessing)

    return data_df


def reshape_numpy_array(data_series: pd.Series) -> np.ndarray:
    """Convert pd.Series to numpy array and reshape it too"""
    return np.asarray(data_series.to_list()).reshape(-1, *IMG_SHAPE)


def apply_image_data_preprocessing(x_data_df: pd.DataFrame, image_feature_name: str) -> np.ndarray:
    """First we apply the transformation that takes a base64 encoded string to a pixel array, then we
    shape that numpy array into the appropriate form for our image size (224 x 224)
    """
    X_data_df = preprocessing_X_transform(x_data_df, image_feature_name)
    X_data = reshape_numpy_array(X_data_df[image_feature_name])
    return X_data


def make_X_transformer_pipeline(X: pd.DataFrame, image_col) -> Pipeline:
    """Image preprocessing pipeline"""

    img_preprocessing_transformer = Pipeline(
        steps=[
            (
                "apply_preprocessing",
                FunctionTransformer(
                    apply_image_data_preprocessing,
                    validate=False,
                    kw_args={"image_feature_name": image_col},
                ),
            ),
        ],
        verbose=True,
    )
    return img_preprocessing_transformer


def fit_image_classifier_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    class_order: List[str],
    image_col="image",
) -> Model:
    """DataRobot stores images as base64 strings. So this function will use a pipeline to convert the base64 string to
    an array of pixels. We use a pipeline because we will need to apply the same transformation both while the model
    is training, i.e. during the fit() hook in custom.py, but also when we are predicting on new data, i.e.
    the predict() hook in custom.py. We will also split the data internally to allow early stopping and
    other neural network best practices.
    """

    # Split the training data into an INTERNAL train / validation split. This is common in
    # neural networks to provide early stopping, etc. while training
    X_train, X_valid, y_train, y_valid = get_transformed_train_validation_split(X, y, class_order)
    X_transformer = make_X_transformer_pipeline(X_train, image_col)

    # Note that DataRobot provides its own easy, no-code image augmentation functionality
    # This is used for convenience to provide generators and allow (optionally) additional
    # image augmentation that may not be available in DataRobot
    X_train_transformed = X_transformer.fit_transform(X_train, y_train)
    train_gen = get_image_augmentation_gen(X_train_transformed, y_train, BATCH_SIZE, SEED)

    X_valid_transformed = X_transformer.fit_transform(X_valid, y_valid)
    valid_gen = get_image_augmentation_gen(X_valid_transformed, y_valid, BATCH_SIZE, SEED)

    train_features, train_labels = extract_features_from_pretrained_network(
        train_gen, len(X_train), get_pretrained_base_model()
    )
    valid_features, valid_labels = extract_features_from_pretrained_network(
        valid_gen, len(X_valid), get_pretrained_base_model()
    )

    # Note: the first channel is the number of samples in the dataset that we transformed
    clf_model = create_image_binary_classification_model(train_features.shape[1:])
    clf_model.fit(
        x=train_features,
        y=train_labels,
        steps_per_epoch=train_gen.n // train_gen.batch_size,
        epochs=EPOCHS,
        validation_data=(valid_features, valid_labels),
        validation_steps=valid_gen.n // valid_gen.batch_size,
        callbacks=get_all_callbacks(),
    )

    X_transformer.steps.append(["model", clf_model])
    return X_transformer


def serialize_estimator_pipeline(estimator_pipeline: Pipeline, output_dir: str) -> None:
    """
    Save the estimator pipeline object in the file path provided.
    This function extracts the preprocessor pipeline and estimator model from the pipeline
    object and construct dict (w/ keys 'preprocessor_pipeline' & 'model') to save as a joblib
    file in the given path. This is required as there is no default save method currently
    available. Also with this we consolidate and bundle the preprocessor pipeline and keras model
    as one single joblib file, which otherwise would require two different file '.pkl' and '.h5'

    Parameters
    ----------
    estimator_pipeline: Pipeline
        Estimator pipeline object with necessary preprocessor and estimator(classifier/regressor) included.
    output_dir: str
        Output directory path where the joblib would be saved/dumped.

    Returns
    -------
    Nothing
    """
    # extract keras model from the pipeline obj and use native serialization
    keras_model = estimator_pipeline[-1]
    tensorflow.keras.models.save_model(keras_model, Path(output_dir) / "model.h5")

    # extract and save the preprocessor
    preprocessor = estimator_pipeline[:-1]
    output_file_path = Path(output_dir) / "artifact.joblib"
    joblib.dump(preprocessor, output_file_path)


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
    # Load the preprocessor object
    joblib_file_path = Path(input_dir) / "artifact.joblib"
    prep_pipeline = joblib.load(joblib_file_path)
    prep_pipeline.steps.append(
        [
            "feature_extraction",
            FunctionTransformer(get_pretrained_base_model().predict, validate=False),
        ]
    )
    # Load the keras model
    keras_model = load_model(Path(input_dir) / "model.h5")

    # Rebuild the original pipeline
    pipeline = Pipeline([("preprocessor", prep_pipeline), ("estimator", keras_model)], verbose=True)
    return pipeline
