from pathlib import Path
import logging
import os
import pickle
import sys
import textwrap

import numpy as np
import pandas as pd
import sklearn

from datarobot_drum.drum.artifact_predictors.keras_predictor import KerasPredictor
from datarobot_drum.drum.artifact_predictors.pmml_predictor import PMMLPredictor
from datarobot_drum.drum.artifact_predictors.sklearn_predictor import SKLearnPredictor
from datarobot_drum.drum.artifact_predictors.torch_predictor import PyTorchPredictor
from datarobot_drum.drum.artifact_predictors.xgboost_predictor import XGBoostPredictor
from datarobot_drum.drum.common import (
    CUSTOM_FILE_NAME,
    CustomHooks,
    LOGGER_NAME_PREFIX,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    REGRESSION_PRED_COLUMN,
)
from datarobot_drum.drum.custom_fit_wrapper import MAGIC_MARKER
from datarobot_drum.drum.exceptions import DrumCommonException


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
            PMMLPredictor(),
            SKLearnPredictor(),
        ]
        self._predictor_to_use = None
        self._custom_hooks = {}
        self._model = None
        self._model_dir = model_dir

        for hook in CustomHooks.ALL:
            self._custom_hooks[hook] = None

    def load_custom_hooks(self):
        custom_file_paths = list(Path(self._model_dir).rglob("{}.py".format(CUSTOM_FILE_NAME)))
        assert len(custom_file_paths) <= 1

        if len(custom_file_paths) == 0:
            print("No {}.py file detected in {}".format(CUSTOM_FILE_NAME, self._model_dir))
            return

        custom_file_path = custom_file_paths[0]
        print("Detected {} .. trying to load hooks".format(custom_file_path))
        sys.path.insert(0, os.path.dirname(custom_file_path))

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

        try:
            model = self._custom_hooks[CustomHooks.LOAD_MODEL](self._model_dir)
        except Exception as exc:
            raise type(exc)(
                "Model loading hook failed to load model: {}".format(exc)
            ).with_traceback(sys.exc_info()[2]) from None

        if not model:
            raise DrumCommonException("Model loading hook failed to load model")

        self._logger.debug("Model was successfully loaded by load hook")
        return model

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
            files_list = os.listdir(self._model_dir)
            files_list_str = " | ".join(files_list)
            raise DrumCommonException(
                "\n\nCould not find model artifact file in: {} supported by default predictors.\n"
                "They support filenames with the following extensions {}.\n"
                "If your artifact is not supported by default predictor, implement custom.load_model hook.\n"
                "List of files got here are: {}".format(
                    self._model_dir, list(all_supported_extensions), files_list_str
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
                try:
                    model = pred.load_model_from_artifact(model_artifact_file)
                except Exception as exc:
                    raise type(exc)(
                        "Could not load model from artifact file: {}".format(exc)
                    ).with_traceback(sys.exc_info()[2]) from None
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
        if not isinstance(to_validate, (pd.DataFrame, np.ndarray)):
            raise ValueError(
                "{} must return a DataFrame; but received {}".format(hook, type(to_validate))
            )

    def _validate_predictions(self, to_validate, positive_class_label, negative_class_label):
        self._validate_data(to_validate, "Predictions")
        columns_to_validate = set(list(to_validate.columns))
        if positive_class_label and negative_class_label:
            if columns_to_validate != {positive_class_label, negative_class_label}:
                raise ValueError(
                    "Expected predictions to have columns {}, but encountered {}".format(
                        {positive_class_label, negative_class_label}, columns_to_validate
                    )
                )
            try:
                added_probs = [
                    a + b
                    for a, b in zip(
                        to_validate[positive_class_label], to_validate[negative_class_label]
                    )
                ]
                np.testing.assert_almost_equal(added_probs, 1)
            except AssertionError:
                raise ValueError("Your prediction probabilities do not add up to 1.")

        elif columns_to_validate != {REGRESSION_PRED_COLUMN}:
            raise ValueError(
                "Expected predictions to have a single {} column, but encountered {}".format(
                    REGRESSION_PRED_COLUMN, columns_to_validate
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
            try:
                # noinspection PyCallingNonCallable
                data = self._custom_hooks[CustomHooks.TRANSFORM](data, model)
            except Exception as exc:
                raise type(exc)(
                    "Model transform hook failed to transform dataset: {}".format(exc)
                ).with_traceback(sys.exc_info()[2]) from None
            self._validate_data(data, CustomHooks.TRANSFORM)

        positive_class_label = kwargs.get(POSITIVE_CLASS_LABEL_ARG_KEYWORD, None)
        negative_class_label = kwargs.get(NEGATIVE_CLASS_LABEL_ARG_KEYWORD, None)

        if self._custom_hooks.get(CustomHooks.SCORE):
            try:
                # noinspection PyCallingNonCallable
                predictions = self._custom_hooks[CustomHooks.SCORE](data, model, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    "Model score hook failed to make predictions: {}".format(exc)
                ).with_traceback(sys.exc_info()[2]) from None
        else:
            try:
                predictions = self._predictor_to_use.predict(data, model, **kwargs)
            except Exception as exc:
                raise type(exc)("Failure when making predictions: {}".format(exc)).with_traceback(
                    sys.exc_info()[2]
                ) from None

        if self._custom_hooks.get(CustomHooks.POST_PROCESS):
            try:
                # noinspection PyCallingNonCallable
                predictions = self._custom_hooks[CustomHooks.POST_PROCESS](predictions, model)
            except Exception as exc:
                raise type(exc)(
                    "Model post-process hook failed to post-process predictions: {}".format(exc)
                ).with_traceback(sys.exc_info()[2]) from None

        self._validate_predictions(predictions, positive_class_label, negative_class_label)

        return predictions

    def _drum_autofit_internal(self, X, y, output_dir):
        """
        A user can surround an sklearn pipeline or estimator with the drum_autofit() function,
        importable from drum, which will tag the object that is passed in with a magic variable.
        This function searches thru all the pipelines and estimators imported from all the modules
        in the code directory, and looks for this magic variable. If it finds it, it will
        load the object here, and call fit on it. Then, it will serialize the fit model out
        to the output directory. If it can't find the wrapper, it will return False, if it
        successfully runs fit, it will return True, otherwise it will throw a DrumCommonException.

        Returns
        -------
        Boolean, whether fit was run
        """
        model_dir_limit = 100
        marked_object = None
        files_in_model_dir = list(Path(self._model_dir).rglob("*.py"))
        if len(files_in_model_dir) == 0:
            return False
        if len(files_in_model_dir) > model_dir_limit:
            self._logger.warning(
                "There are more than {} files in this directory".format(model_dir_limit)
            )
            return False
        for filepath in files_in_model_dir:
            filename = os.path.basename(filepath)
            sys.path.insert(0, os.path.dirname(filepath))
            try:
                module = __import__(filename[:-3])
            except ImportError as e:
                self._logger.warning(
                    "File at path {} could not be imported: {}".format(filepath, str(e))
                )
                continue
            for object_name in dir(module):
                _object = getattr(module, object_name)
                if isinstance(_object, sklearn.base.BaseEstimator):
                    if hasattr(_object, MAGIC_MARKER):
                        marked_object = _object
                        break

        if marked_object is not None:
            marked_object.fit(X, y)
            with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
                pickle.dump(marked_object, fp)
            return True
        return False

    def fit(self, X, y, output_dir, class_order=None, row_weights=None):
        if self._custom_hooks.get(CustomHooks.FIT):
            self._custom_hooks[CustomHooks.FIT](
                X, y, output_dir, class_order=class_order, row_weights=row_weights
            )
        elif self._drum_autofit_internal(X, y, output_dir):
            return
        else:
            hooks = [
                "{}: {}".format(hook, fn is not None) for hook, fn in self._custom_hooks.items()
            ]
            raise DrumCommonException(
                "\nfit() method must be implemented in a file named 'custom.py' in the provided code_dir: '{}' \n"
                "Here is a list of files in this dir. {}\n"
                "Here are the hooks your custom.py file has: {}".format(
                    self._model_dir, os.listdir(self._model_dir)[:100], hooks
                )
            )
