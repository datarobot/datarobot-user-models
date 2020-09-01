from typing import Tuple

# keras imports
import keras
from keras.models import Sequential, Model
from keras.layers import Dense  # core layers
from keras.layers import GlobalAveragePooling2D  # CNN layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

# scikit-learn imports
from sklearn.preprocessing import LabelBinarizer, label_binarize, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# pandas/numpy imports
import pandas as pd
import numpy as np

import joblib
import io
import base64
import h5py
from PIL import Image
from pathlib import Path

from typing import List, Iterator


# define constants

IMG_SIZE = 150
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
EPOCHS = 100
BATCH_SIZE = 32
TEST_SIZE = 0.33
SEED = 4321
NUM_CLASSES = 2


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


def get_imputation_img() -> str:
    """ black image in base64 str for data imputation filling """
    black_PIL_img = Image.fromarray(np.zeros(IMG_SHAPE, dtype="float32"), "RGB")
    return get_base64_str_from_PIL_img(black_PIL_img)


def extract_features(
    generator: Iterator[tuple], sample_count: int, base_model: Model
) -> Tuple[np.ndarray, np.ndarray]:
    """ extract the features using the CNN base model """
    conv_base_actvn_output_shape = tuple(base_model.layers[-1].output.shape[1:])
    features = np.zeros((sample_count, *conv_base_actvn_output_shape))
    labels = np.zeros((sample_count))
    i = 0

    for inputs_batch, labels_batch in generator:
        # using base model to predict and extract the features
        features_batch = base_model.predict(inputs_batch)
        features[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] = labels_batch.squeeze()
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break

    return features, labels


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


def get_all_callbacks() -> List[keras.callbacks.Callback]:
    """ List of all keras callbacks """
    es = EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="auto", min_delta=1e-4)
    return [es]


def apply_image_data_preprocessing(x_data_df: pd.DataFrame, image_feature_name: str) -> np.ndarray:
    """ Image data preprocessing before fit """
    X_data_df = preprocessing_X_transform(x_data_df, image_feature_name)
    X_data = reshape_numpy_array(X_data_df[image_feature_name])
    return X_data


def get_image_augmentation_gen(X_data, y_data, bs, seed) -> Iterator[tuple]:
    """ Generator which yields tuple of image data and corresponding labels in np.array """
    # normalize by rescaling
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    datagen.fit(X_data)

    image_aug_generator = datagen.flow(x=X_data, y=y_data, batch_size=bs, seed=seed)
    return image_aug_generator


def get_pretrained_base_model() -> Model:
    """ A base pretrained model to build on top of """
    pretrained_model = MobileNetV2(include_top=False, input_shape=IMG_SHAPE, classes=NUM_CLASSES,)
    pretrained_model.trainable = False
    return pretrained_model


def create_image_binary_classification_model() -> Model:
    """
    Create an image binary classification model.

    Returns
    -------
    Model
        Compiled binary classification model
    """
    model = Sequential()
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["binary_accuracy"]
    )
    return model


def get_transformed_train_test_split(
    X_df: pd.DataFrame, y_series: pd.Series, class_order: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ split train/test data after apply label encoder on y """
    assert len(X_df) == len(y_series)
    # preprocessing steps
    if class_order:
        y_series = label_binarize(y_series, classes=class_order)
    else:
        lb = LabelBinarizer()
        y_series = lb.fit_transform(y_series)

    # convert np.ndarray to pd.Series
    y_series = pd.Series(y_series.squeeze())

    # split train/test data
    msk = np.random.rand(len(X_df)) < TEST_SIZE
    x_test_data = X_df[msk]
    x_train_data = X_df[~msk]

    y_test_data = y_series[msk]
    y_train_data = y_series[~msk]

    return (x_train_data, x_test_data, y_train_data, y_test_data)


def convert_np_to_df(np_array, img_col) -> pd.DataFrame:
    """ simple utility to convert numpy array to dataframe """
    return pd.DataFrame(data=np_array, columns=[img_col])


def get_image_feature_column(X: pd.DataFrame) -> str:
    """ Get only the image feature column name from X """
    X = X.select_dtypes(object)
    img_features_col_mask = [X[col].str.startswith("/9j/", na=False).any() for col in X]

    # should have just one image feature
    assert sum(img_features_col_mask) == 1, "expecting just one image feature column"

    img_col = X.columns[np.argmax(img_features_col_mask)]
    return img_col


def make_X_transformer_pipeline(X: pd.DataFrame) -> Pipeline:
    """ Image preprocessing pipeline """
    img_col = get_image_feature_column(X)

    img_preprocessing_transformer = Pipeline(
        steps=[
            ("img_imputer", SimpleImputer(fill_value=get_imputation_img(), strategy="constant")),
            (
                "np_to_df",
                FunctionTransformer(convert_np_to_df, validate=False, kw_args={"img_col": img_col}),
            ),
            (
                "apply_preprocessing",
                FunctionTransformer(
                    apply_image_data_preprocessing,
                    validate=False,
                    kw_args={"image_feature_name": img_col},
                ),
            ),
        ],
        verbose=True,
    )
    return img_preprocessing_transformer


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
    # extract preprocessor pipeline
    preprocessor = estimator_pipeline[:-1]

    # extract keras model from the pipeline obj
    keras_model = estimator_pipeline[-1]

    # save the model (in '.h5' format - as its easy to load using keras' load_model() later)
    # to BytesIO obj (in RAM - as its easy to joblib dump later)
    io_container = io.BytesIO()
    with h5py.File(io_container, mode="w") as file:
        # setting 'include_optimizer' to True is only needed when warm starting.
        # Will save a lot of space as well
        keras_model.save(file, include_optimizer=False)

    # save the preprocessor and the model to dictionary
    model_dict = dict()
    model_dict["preprocessor_pipeline"] = preprocessor
    model_dict["model"] = io_container

    # save the dict obj as a joblib file
    output_file_path = Path(output_dir) / "artifact.joblib"
    joblib.dump(model_dict, output_file_path)


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


def fit_image_classifier_pipeline(
    X: pd.DataFrame, y: pd.Series, class_order: List[str]
) -> Pipeline:
    """ Fit the estimator pipeline """
    X_train, X_test, y_train, y_test = get_transformed_train_test_split(X, y, class_order)
    X_transformer = make_X_transformer_pipeline(X_train)

    X_train_transformed = X_transformer.fit_transform(X_train, y_train)
    train_gen = get_image_augmentation_gen(X_train_transformed, y_train, BATCH_SIZE, SEED)

    X_test_transformed = X_transformer.fit_transform(X_test, y_test)
    test_gen = get_image_augmentation_gen(X_test_transformed, y_test, BATCH_SIZE, SEED)

    train_features, train_labels = extract_features(
        train_gen, len(X_train_transformed), get_pretrained_base_model()
    )
    test_features, test_labels = extract_features(
        test_gen, len(X_test_transformed), get_pretrained_base_model()
    )

    clf_model = create_image_binary_classification_model()
    clf_model.fit(
        x=train_features,
        y=train_labels,
        steps_per_epoch=train_gen.n // train_gen.batch_size,
        epochs=EPOCHS,
        validation_data=(test_features, test_labels),
        validation_steps=test_gen.n // test_gen.batch_size,
        callbacks=get_all_callbacks(),
    )

    # append few more steps to the preprocess-pipeline for the final prediction
    X_transformer.steps.append(
        ["preprocess_input", FunctionTransformer(pretrained_preprocess_input, validate=False)]
    )
    X_transformer.steps.append(
        [
            "feature_extraction",
            FunctionTransformer(get_pretrained_base_model().predict, validate=False),
        ]
    )

    # append "model" to the pipeline
    X_transformer.steps.append(["model", clf_model])
    return X_transformer
