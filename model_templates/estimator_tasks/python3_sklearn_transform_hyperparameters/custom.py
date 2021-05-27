import pickle
import pandas as pd
from scipy.sparse.csr import csr_matrix

from create_transform_pipeline import make_pipeline


def fit(
    X: pd.DataFrame, y: pd.Series, output_dir: str, parameters: dict, **kwargs,
):
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom tasks. For custom transformers, the
    transform hook below is also required.

    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X: pd.DataFrame - estimator_tasks data to perform fit on
    y: pd.Series - target data to perform fit on
    output_dir: the path to write output. This is the path provided in '--output' parameter of the
        'drum fit' command.
    parameters: dict
        A dictionary of parameters defined within the model-metadata.yaml file.
    kwargs: Added for forwards compatibility

    Returns
    -------
    Nothing
    """
    transformer = make_pipeline()

    if not parameters:
        raise ValueError("Did not receive parameters")

    # Parameters are provided during fit as a dict with names set according to your model-metadata.yaml file.
    # In this example, we set the various transformer parameters in our sklearn pipeline.
    transformer.set_params(
        **{
            "num__imputer__strategy": parameters["numeric_imputer_strategy"],
            "num__scaler__with_mean": parameters["numeric_standardize_with_mean"],
            "cat__imputer__fill_value": parameters["categorical_fill"],
        }
    )

    # Only set numeric imputer's fill value if the strategy is constant
    if parameters["numeric_imputer_strategy"] == "constant":
        transformer.set_params(num__imputer__fill_value=parameters["numeric_imputer_constant_fill"])

    transformer.fit(X, y)

    # You must serialize out your transformer to the output_dir given, however if you wish to change this
    # code, you will probably have to add a load_model method to read the serialized model back in
    # When prediction is done.
    # Check out this doc for more information on serialization https://github.com/datarobot/custom-\
    # model-templates/tree/master/custom_model_runner#python
    # NOTE: We currently set a 10GB limit to the size of the serialized model or transformer
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(transformer, fp)


def transform(X, transformer, y=None):
    """
    Parameters
    ----------
    X: pd.DataFrame - estimator_tasks data to perform transform on
    transformer: object - trained transformer object
    y: pd.Series (optional) - target data to perform transform on
    Returns
    -------
    transformed DataFrame resulting from applying transform to incoming data
    """
    transformed = transformer.transform(X)
    if type(transformed) == csr_matrix:
        return pd.DataFrame.sparse.from_spmatrix(transformed)
    else:
        return pd.DataFrame(transformed)
