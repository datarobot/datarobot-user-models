import pickle
import pandas as pd
import numpy as np

from create_transform_pipeline import make_pipeline


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom training models.
    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X: pd.DataFrame - training data to perform fit on
    y: pd.Series - target data to perform fit on
    output_dir: the path to write output. This is the path provided in '--output' parameter of the
        'drum fit' command.
    kwargs: Added for forwards compatibility

    Returns
    -------
    Nothing
    """
    transformer = make_pipeline()
    transformer.fit(X, y)

    # You must serialize out your model to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(transformer, fp)


def transform(X, y, model):
    # use preprocessing as trained in fit, and log-transform y var
    return pd.DataFrame(model.transform(X)), np.log(y+.01)
