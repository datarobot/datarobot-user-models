from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm as maxnorm
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd

import joblib
import io
import h5py
from pathlib import Path


def create_regression_model(num_features: int) -> Sequential:
    """
    Create a regression model.

    Parameters
    ----------
    num_features: int
        Number of features in X to be trained with

    Returns
    -------
    model: Sequential
        Compiled regression model
    """
    input_dim, output_dim = num_features, 1

    # create model
    model = Sequential(
        [
            Dense(input_dim, activation="relu", input_dim=input_dim, kernel_initializer="normal"),
            Dense(input_dim // 2, activation="relu", kernel_initializer="normal"),
            Dense(output_dim, kernel_initializer="normal"),
        ]
    )
    model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])
    return model


def create_classification_model(num_features: int, num_labels: int) -> Sequential:
    """
    Create a binary classification model.

    Parameters
    ----------
    num_features: int
        Number of features in X to be trained with

    Returns
    -------
    model: Sequential
        Compiled binary classification model
    """
    input_dim, output_dim = num_features, num_labels if num_labels > 2 else 1
    if num_labels > 2:
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
        final_activation = "softmax"
    else:
        loss = "binary_crossentropy"
        metrics = ["binary_accuracy"]
        final_activation = "sigmoid"

    # create model
    model = Sequential()

    # input layer
    model.add(
        Dense(num_features, input_dim=input_dim, activation="relu", kernel_constraint=maxnorm(3))
    )

    # hidden layer
    hidden_units = (input_dim * 2) + 1
    model.add(Dense(hidden_units, activation="relu", kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(output_dim, activation=final_activation))

    # Compile model
    model.compile(loss=loss, optimizer="adam", metrics=metrics)
    return model


def make_classifier_pipeline(X: pd.DataFrame, num_labels: int) -> Pipeline:
    """
    Make the classifier pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X: pd.DataFrame
        X containing all the required features for training
    num_labels: int
        The number of output labels

    Returns
    -------
    classifier_pipeline: Pipeline
        Classifier pipeline with preprocessor and estimator
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    # exclude any completely-missing columns when checking for numerics
    num_features = list(X.dropna(axis=1, how="all").select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # create model
    estimator = KerasClassifier(
        build_fn=create_classification_model,
        num_features=len(num_features),
        num_labels=num_labels,
        epochs=30,
        batch_size=8,
        verbose=1,
    )

    # pipeline with preprocessor and estimator bundled
    classifier_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return classifier_pipeline


def make_regressor_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Make the regressor pipeline with the required preprocessor steps and estimator in the end.

    Parameters
    ----------
    X: pd.DataFrame
        X containing all the required features for training

    Returns
    -------
    regressor_pipeline: Pipeline
        Regressor pipeline with preprocessor and estimator
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_features = list(X.dropna(axis=1, how="all").select_dtypes(include=numerics).columns)

    # This example model only uses numeric features and drops the rest
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("standardize", StandardScaler())]
    )
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features)])

    # create model
    estimator = KerasRegressor(
        build_fn=create_regression_model,
        num_features=len(num_features),
        epochs=200,
        batch_size=8,
        verbose=1,
    )

    # pipeline with preprocessor and estimator bundled
    regressor_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return regressor_pipeline


def serialize_estimator_pipeline(estimator_pipeline: Pipeline, output_dir_path: str) -> None:
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

    output_dir_path: str
        Output directory path where the joblib would be saved/dumped.

    Returns
    -------
    Nothing
    """
    # extract preprocessor pipeline
    preprocessor = estimator_pipeline.named_steps.preprocessor

    # extract keras model from the pipeline obj
    keras_model = estimator_pipeline.named_steps.estimator.model

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
    output_file_path = Path(output_dir_path) / "artifact.joblib"
    joblib.dump(model_dict, output_file_path)


def deserialize_estimator_pipeline(input_dir: str) -> Pipeline:
    """
    Load estimator pipeline from the given joblib file.

    Parameters
    ----------
    input_dir: str
        The directory from which joblib file would be loaded.

    Returns
    -------
    pipeline: Pipeline
        Constructed pipeline with necessary preprocessor steps and estimator to predict/score.
    """
    # load the dictionary obj from the joblib file
    joblib_file_path = Path(input_dir) / "artifact.joblib"
    estimator_dict = joblib.load(joblib_file_path)
    with h5py.File(estimator_dict["model"], mode="r") as fp:
        keras_model = load_model(fp)

    pipeline = Pipeline(
        [("preprocessor", estimator_dict["preprocessor_pipeline"]), ("estimator", keras_model)]
    )
    return pipeline
