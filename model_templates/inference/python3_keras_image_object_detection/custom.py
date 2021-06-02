from typing import Any

from sklearn.pipeline import Pipeline
import json

from model_load_utils import (
    load_image_object_detection_inference_pipeline,
    predict_with_preprocessing,
)


def load_model(input_dir: str) -> Pipeline:
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    This keras estimator requires 'load_model()' to be overridden. Coz as it involves pipeline of
    preprocessor and estimator bundled together, it requires a special handling (oppose to usually
    simple keras.models.load_model() or unpickling) to load the model. Currently there is no elegant
    default method to save the keras classifier/regressor along with the sklearn pipeline. Hence we
    use deserialize_estimator_pipeline() to load the model pipeline to predict.

    Parameters
    ----------
    input_dir: str

    Returns
    -------
    pipelined_model: Pipeline
        Estimator pipeline obj
    """
    model = load_image_object_detection_inference_pipeline(input_dir)
    return model


def transform(b64_image_array: str, model: Any) -> list:
    """
    Intended to apply transformations to the prediction data before making predictions. This is
    most useful if DRUM supports the model's library, but your model requires additional data
    processing before it can make predictions

    Parameters
    ----------
    data : is the dataframe given to DRUM to make predictions on
    model : is the deserialized model loaded by DRUM or by `load_model`, if supplied

    Returns
    -------
    Transformed data: np.ndarray
    """
    predicted_labels = predict_with_preprocessing(model, b64_image_array)
    return predicted_labels


def score_unstructured(model, data, query, **kwargs):
    print("Model: ", model)
    print("Incoming content type params: ", kwargs)
    print("Incoming data type: ", type(data))
    print("Incoming data: ", data)

    print("Incoming query params: ", query)
    if isinstance(data, bytes):
        data = data.decode("utf8")
    ret = transform(data, model).astype(int).tolist()

    ret_mode = query.get("ret_mode", "")
    if ret_mode == "binary":
        ret_data = ret.tobytes()
        ret_kwargs = {"mimetype": "application/octet-stream"}
        ret = ret_data, ret_kwargs
    else:
        ret = json.dumps(ret)
    return ret


# def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
#     """
#     This hook is only needed if you would like to use DRUM with a framework not natively
#     supported by the tool.
#
#     Parameters
#     ----------
#     data : is the dataframe to make predictions against. If `transform` is supplied,
#     `data` will be the transformed data.
#     model : is the deserialized model loaded by DRUM or by `load_model`, if supplied
#     kwargs : additional keyword arguments to the method
#     In case of classification model class labels will be provided as the following arguments:
#     - `positive_class_label` is the positive class label for a binary classification model
#     - `negative_class_label` is the negative class label for a binary classification model
#
#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Binary Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row should sum to 1.0
#       Regression: must have a single column called `Predictions` with numerical values
#
#     """

# def post_process(predictions: pd.DataFrame, model: Any) -> pd.DataFrame:
#     """
#     This method is only needed if your model's output does not match the above expectations
#
#     Parameters
#     ----------
#     predictions : is the dataframe of predictions produced by DRUM or by
#       the `score` hook, if supplied
#     model : is the deserialized model loaded by DRUM or by `load_model`, if supplied
#
#     Returns
#     -------
#     This method should return predictions as a dataframe with the following format:
#       Binary Classification: must have columns for each class label with floating- point class
#         probabilities as values. Each row
#     should sum to 1.0
#       Regression: must have a single column called `Predictions` with numerical values
#
#     """
