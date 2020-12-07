import pickle
import pandas as pd
from scipy.sparse.csr import csr_matrix

from create_transform_pipeline import make_pipeline


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom training models. For custom transformers, the
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
    transformer = make_pipeline()
    transformer.fit(X, y)

    # You must serialize out your transformer to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model or transformer
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(transformer, fp)


def transform(X, y, transformer):
    """
    Parameters
    ----------
    X: pd.DataFrame - training data to perform transform on
    y: pd.Series - target data to perform transform on
    transformer: object - trained transformer object

    Returns
    -------
    transformed DataFrame resulting from applying transform to incoming data
    """
    transformed = transformer.transform(X)
    if type(transform) == csr_matrix:
        return pd.DataFrame.sparse.from_spmatrix(transformed), y
    else:
        return pd.DataFrame(transformed), y
