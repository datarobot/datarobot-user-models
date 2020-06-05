import os

import pandas as pd
import logging
import sys
import textwrap

from datarobot_drum.drum.common import (
    CustomHooks,
    CUSTOM_FILE_NAME,
    LOGGER_NAME_PREFIX,
    REGRESSION_PRED_COLUMN,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
)

from datarobot_drum.drum.exceptions import DrumCommonException

from datarobot_drum.drum.artifact_predictors.artifact_predictor import (
    KerasPredictor,
    SKLearnPredictor,
    PyTorchPredictor,
    XGBoostPredictor,
)


class PythonModelAdapter:
    def __init__(self, model_dir):
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)

        # Get all the artifact predictors we have
        # let `SKLearnPredictor` be the last item, as we iterate through this list to find the
        # predictor for the given model artifact (based on the instance type of the estimator) it might
        # overlap with other predictors especially the ones with `sklearn.pipeline`
        self._artifact_predictors = [
            KerasPredictor(),
            XGBoostPredictor(),
            PyTorchPredictor(),
            SKLearnPredictor(),
        ]
        self._predictor_to_use = None
        self._custom_hooks = {}
        self._model = None
        self._model_dir = model_dir

        for hook in CustomHooks.ALL:
            self._custom_hooks[hook] = None

    def load_custom_hooks(self):
        custom_file_path = os.path.join(self._model_dir, CUSTOM_FILE_NAME + ".py")

        if not os.path.isfile(custom_file_path):
            self._logger.debug("No {} detected".format(custom_file_path))
            return

        self._logger.debug("Detected {} .. trying to load hooks".format(custom_file_path))
        sys.path.append(self._model_dir)

        try:
            custom_module = __import__(CUSTOM_FILE_NAME)

            for hook in CustomHooks.ALL:
                self._custom_hooks[hook] = getattr(custom_module, hook, None)

            if self._custom_hooks.get(CustomHooks.INIT):
                # noinspection PyCallingNonCallable
                self._custom_hooks[CustomHooks.INIT](code_dir=self._model_dir)

            self._logger.debug("Hooks loaded: {}".format(self._custom_hooks))
        except ImportError as e:
            self._logger.error("Could not load hooks: {}".format(e))
            raise DrumCommonException(
                "Failed loading hooks from [{}] : {}".format(custom_file_path, e)
            )

    def load_model_from_artifact(self):
        """
        Load the serialized model from it's artifact.
        Returns
        -------
        Any
            The deserialized model
        Raises
        ------
        DrumCommonException if model loading failed.
        """
        if self._custom_hooks[CustomHooks.LOAD_MODEL]:
            self._model = self._load_model_via_hook()
        else:
            model_artifact_file = self._detect_model_artifact_file()
            self._model = self._load_via_predictors(model_artifact_file)

        # If a score hook is not given we need to find a predictor that can handle this model
        if not self._custom_hooks[CustomHooks.SCORE]:
            self._find_predictor_to_use()

        return self._model

    def _load_model_via_hook(self):
        self._logger.debug("Load model hook will be used to load the model")
        # noinspection PyCallingNonCallable
        model = self._custom_hooks[CustomHooks.LOAD_MODEL](self._model_dir)
        if model:
            self._logger.debug("Model was successfully loaded by load hook")
            return model
        else:
            raise DrumCommonException("Model loading hook failed to load model")

    def _detect_model_artifact_file(self):
        # No model was loaded - so there is no local hook - so we are using our artifact predictors
        all_supported_extensions = set(p.artifact_extension for p in self._artifact_predictors)
        self._logger.debug("Supported suffixes: {}".format(all_supported_extensions))
        model_artifact_file = None

        for filename in os.listdir(self._model_dir):
            path = os.path.join(self._model_dir, filename)
            if os.path.isdir(path):
                continue

            if any(filename.endswith(extension) for extension in all_supported_extensions):
                if model_artifact_file:
                    raise DrumCommonException(
                        "Multiple serialized model files found. Remove extra artifacts "
                        "or overwrite custom.load_model()"
                    )
                model_artifact_file = path

        if not model_artifact_file:
            raise DrumCommonException(
                "Could not find model artifact file in: {} supported by default predictors. "
                "They support filenames with the following extensions {}. "
                "If your artifact is not supported by default predictor, implement custom.load_model hook".format(
                    self._model_dir, list(all_supported_extensions)
                )
            )

        self._logger.debug("model_artifact_file: {}".format(model_artifact_file))
        return model_artifact_file

    def _load_via_predictors(self, model_artifact_file):

        model = None
        pred_that_support_artifact = []
        for pred in self._artifact_predictors:
            if pred.is_artifact_supported(model_artifact_file):
                pred_that_support_artifact.append(pred)

            if pred.can_load_artifact(model_artifact_file):
                model = pred.load_model_from_artifact(model_artifact_file)
                break

        if not model:
            if len(pred_that_support_artifact) > 0:
                framework_err = """
                    The following frameworks support this model artifact
                    but could not load the model. Check if requirements are missing
                """

                for pred in pred_that_support_artifact:
                    framework_err += "Framework: {}, requirements: {}".format(
                        pred.name, pred.framework_requirements()
                    )

                raise DrumCommonException(textwrap.dedent(framework_err))
            else:
                raise DrumCommonException(
                    "Could not load model from artifact file {}."
                    " No builtin support for this model was detected".format(model_artifact_file)
                )

        self._model = model
        return model

    def _find_predictor_to_use(self):
        self._predictor_to_use = None
        for pred in self._artifact_predictors:
            if pred.can_use_model(self._model):
                self._predictor_to_use = pred
                break

        if not self._predictor_to_use and not self._custom_hooks[CustomHooks.SCORE]:
            raise DrumCommonException(
                "Could not find any framework to handle loaded model and a {} "
                "hook is not provided".format(CustomHooks.SCORE)
            )

        self._logger.debug("Predictor to use: {}".format(self._predictor_to_use.name))

    @staticmethod
    def _validate_data(to_validate, hook):
        if not isinstance(to_validate, pd.DataFrame):
            raise ValueError(
                "{} must return a DataFrame; but received {}".format(hook, type(to_validate))
            )

    def _validate_predictions(self, to_validate, hook, positive_class_label, negative_class_label):
        self._validate_data(to_validate, hook)
        columns_to_validate = set(list(to_validate.columns))
        if positive_class_label and negative_class_label:
            if columns_to_validate != {positive_class_label, negative_class_label}:
                raise ValueError(
                    "Expected {} predictions to have columns {}, but encountered {}".format(
                        hook, {positive_class_label, negative_class_label}, columns_to_validate
                    )
                )
        elif columns_to_validate != {REGRESSION_PRED_COLUMN}:
            raise ValueError(
                "Expected {} predictions to have a single {} column, but encountered {}".format(
                    hook, REGRESSION_PRED_COLUMN, columns_to_validate
                )
            )

    def predict(self, data, model=None, **kwargs):
        """
        Makes predictions against the model using the custom predict
        method and returns a pandas DataFrame
        If the model is a regression model, the DataFrame will have a single column "Predictions"
        If the model is a classification model, the DataFrame will have a column for each class label
            with their respective probabilities. Positive/negative class labels will be passed in kwargs under
            positive_class_label/negative_class_label keywords.
        Parameters
        ----------
        data: pd.DataFrame
            Data to make predictions against
        model: Any
            The model
        kwargs
        Returns
        -------
        pd.DataFrame
        """
        if self._custom_hooks.get(CustomHooks.TRANSFORM):
            # noinspection PyCallingNonCallable
            data = self._custom_hooks[CustomHooks.TRANSFORM](data, model)
            self._validate_data(data, CustomHooks.TRANSFORM)

        positive_class_label = kwargs.get(POSITIVE_CLASS_LABEL_ARG_KEYWORD, None)
        negative_class_label = kwargs.get(NEGATIVE_CLASS_LABEL_ARG_KEYWORD, None)

        if self._custom_hooks.get(CustomHooks.SCORE):
            # noinspection PyCallingNonCallable
            predictions = self._custom_hooks[CustomHooks.SCORE](data, model, **kwargs)
            self._validate_predictions(
                predictions, CustomHooks.SCORE, positive_class_label, negative_class_label
            )
        else:
            predictions = self._predictor_to_use.predict(data, model, **kwargs)

        if self._custom_hooks.get(CustomHooks.POST_PROCESS):
            # noinspection PyCallingNonCallable
            predictions = self._custom_hooks[CustomHooks.POST_PROCESS](predictions, model)
            self._validate_predictions(
                predictions, CustomHooks.POST_PROCESS, positive_class_label, negative_class_label
            )

        return predictions

    def fit(
        self, X, y, output_dir, class_order=None, row_weights=None,
    ):
        if self._custom_hooks.get(CustomHooks.FIT):
            self._custom_hooks[CustomHooks.FIT](
                X, y, output_dir, class_order=class_order, row_weights=row_weights
            )
        else:
            raise DrumCommonException(
                "fit() method must be implemented in the 'custom.py' in the provided code_dir: '{}'".format(
                    self._model_dir
                )
            )
