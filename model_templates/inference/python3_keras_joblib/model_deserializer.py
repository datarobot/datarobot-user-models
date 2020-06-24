from keras.models import load_model
from sklearn.pipeline import Pipeline
import joblib
import h5py
from pathlib import Path


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
