def load_model(input_dir):
    """
    This hook can be implemented to adjust logic in the scoring mode.

    load_model hook provides a way to implement model loading your self.
    This function should return an object that represents your model. This object will
    be passed to the predict hook for performing predictions.
    This hook can be used to load supported models if your model has multiple artifacts, or
    for loading models that drum does not natively support

    :param input_dir: the directory to load serialized models from
    :returns: Object containing the model - the predict hook will get this object as a parameter
    """

    # Returning a string with value "dummy" as the model.
    return "dummy"


def score_unstructured(model, data, **kwargs):
    return None
