import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit(
    X: pd.DataFrame, y: pd.Series, output_dir: str, **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.
    This hook MUST ALWAYS be implemented for custom tasks. For custom transformers, the
    transform hook below is also required.
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
    transformer = StandardScaler(with_mean=False)
    transformer.fit(X)

    # You must serialize out your transformer to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model or transformer
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(transformer, fp)
