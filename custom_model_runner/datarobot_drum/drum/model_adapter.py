import logging
import os
import pickle
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from inspect import signature

from pathlib import Path

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
    CLASS_LABELS_ARG_KEYWORD,
    REGRESSION_PRED_COLUMN,
    reroute_stdout_to_stderr,
    TargetType,
    PayloadFormat,
    SupportedPayloadFormats,
    StructuredDtoKeys,
    get_pyarrow_module,
)
from datarobot_drum.drum.utils import StructuredInputReadUtils
from datarobot_drum.drum.custom_fit_wrapper import MAGIC_MARKER
from datarobot_drum.drum.exceptions import DrumCommonException

RUNNING_LANG_MSG = "Running environment language: Python."


class PythonModelAdapter:
    def __init__(self, model_dir, target_type=None):
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
        self._custom_hooks = {hook: None for hook in CustomHooks.ALL_PREDICT_FIT_STRUCTURED}
        self._model = None
        self._model_dir = model_dir
        self._target_type = target_type

    @staticmethod
    def _apply_sklearn_transformer(data, model):
        try:
            transformed = model.transform(data)
            if type(transformed) == csr_matrix:
                output_data = pd.DataFrame.sparse.from_spmatrix(transformed)
            else:
                output_data = pd.DataFrame(transformed)
        except Exception as e:
            raise type(e)("Couldn't naively apply transformer:" " {}".format(e)).with_traceback(
                sys.exc_info()[2]
            ) from None
        return output_data

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
            if self._target_type == TargetType.UNSTRUCTURED:
                for hook in CustomHooks.ALL_PREDICT_UNSTRUCTURED:
                    self._custom_hooks[hook] = getattr(custom_module, hook, None)

                if self._custom_hooks[CustomHooks.SCORE_UNSTRUCTURED] is None:
                    raise DrumCommonException(
                        "In '{}' mode hook '{}' must be provided.".format(
                            TargetType.UNSTRUCTURED.value,
                            self._custom_hooks[CustomHooks.SCORE_UNSTRUCTURED],
                        )
                    )
            else:
                for hook in CustomHooks.ALL_PREDICT_FIT_STRUCTURED:
                    self._custom_hooks[hook] = getattr(custom_module, hook, None)

            if self._custom_hooks.get(CustomHooks.INIT):
                # noinspection PyCallingNonCallable
                self._custom_hooks[CustomHooks.INIT](code_dir=self._model_dir)

            self._logger.debug("Hooks loaded: {}".format(self._custom_hooks))
        except ImportError as e:
            self._logger.error("Could not load hooks: {}".format(e))
            raise DrumCommonException(
                "\n\n{}\n"
                "Failed loading hooks from [{}] : {}".format(RUNNING_LANG_MSG, custom_file_path, e)
            )

    def load_model_from_artifact(self):
        """
        Load the serialized model from its artifact.
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
        if (
            self._target_type not in [TargetType.UNSTRUCTURED, TargetType.TRANSFORM]
            and not self._custom_hooks[CustomHooks.SCORE]
        ):
            self._find_predictor_to_use()

        if (
            self._target_type == TargetType.TRANSFORM
            and not self._custom_hooks[CustomHooks.TRANSFORM]
            # don't require transform hook if sklearn transformer
            and "sklearn" not in str(type(self._model))
        ):
            raise DrumCommonException(
                "A transform task requires a user-defined transform hook "
                "for non-sklearn transformers"
            )

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
        all_supported_extensions = list(sorted(all_supported_extensions))
        self._logger.debug("Supported suffixes: {}".format(all_supported_extensions))
        model_artifact_file = None

        for filename in os.listdir(self._model_dir):
            path = os.path.join(self._model_dir, filename)
            if os.path.isdir(path):
                continue

            if any(filename.endswith(extension) for extension in all_supported_extensions):
                if model_artifact_file:
                    raise DrumCommonException(
                        "\n\n{}\n"
                        "Multiple serialized model files found. Remove extra artifacts "
                        "or overwrite custom.load_model()".format(RUNNING_LANG_MSG)
                    )
                model_artifact_file = path

        if not model_artifact_file:
            files_list = sorted(os.listdir(self._model_dir))
            files_list_str = " | ".join(files_list)
            raise DrumCommonException(
                "\n\n{}\n"
                "Could not find model artifact file in: {} supported by default predictors.\n"
                "They support filenames with the following extensions {}.\n"
                "If your artifact is not supported by default predictor, implement custom.load_model hook.\n"
                "List of retrieved files are: {}".format(
                    RUNNING_LANG_MSG, self._model_dir, all_supported_extensions, files_list_str
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

        if model is None:
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
                    "\n\n{}\n"
                    "Could not load model from artifact file {}."
                    " No builtin support for this model was detected".format(
                        RUNNING_LANG_MSG, model_artifact_file
                    )
                )

        self._model = model
        return model

    def _find_predictor_to_use(self):
        # TODO: RAPTOR-4014 need to handle transformers when we don't require transform hook for sklearn pipelines
        self._predictor_to_use = None
        for pred in self._artifact_predictors:
            if pred.can_use_model(self._model):
                self._predictor_to_use = pred
                break

        if not self._predictor_to_use and not self._custom_hooks[CustomHooks.SCORE]:
            raise DrumCommonException(
                "\n\n{}\n"
                "Could not find any framework to handle loaded model and a {} "
                "hook is not provided".format(RUNNING_LANG_MSG, CustomHooks.SCORE)
            )

        self._logger.debug("Predictor to use: {}".format(self._predictor_to_use.name))

    @staticmethod
    def _validate_data(to_validate, hook):
        if not isinstance(to_validate, (pd.DataFrame, np.ndarray)):
            raise ValueError(
                "{} must return a DataFrame; but received {}".format(hook, type(to_validate))
            )

    @staticmethod
    def _validate_transform_rows(data, output_data):
        if data.shape[0] != output_data.shape[0]:
            raise ValueError(
                "Transformation resulted in different number of rows than original data"
            )

    def _validate_predictions(self, to_validate, class_labels):
        self._validate_data(to_validate, "Predictions")
        columns_to_validate = set(str(label) for label in to_validate.columns)
        if class_labels:
            if columns_to_validate != set(str(label) for label in class_labels):
                raise ValueError(
                    "Expected predictions to have columns {}, but encountered {}".format(
                        class_labels, columns_to_validate
                    )
                )
            try:
                added_probs = to_validate.sum(axis=1)
                np.testing.assert_array_almost_equal(added_probs, 1)
            except AssertionError:
                raise ValueError(
                    "Your prediction probabilities do not add up to 1. {}".format(to_validate)
                )

        elif columns_to_validate != {REGRESSION_PRED_COLUMN}:
            raise ValueError(
                "Expected predictions to have a single {} column, but encountered {}".format(
                    REGRESSION_PRED_COLUMN, columns_to_validate
                )
            )

    @property
    def supported_payload_formats(self):
        formats = SupportedPayloadFormats()
        formats.add(PayloadFormat.CSV)
        formats.add(PayloadFormat.MTX)
        pa = get_pyarrow_module()
        if pa is not None:
            formats.add(PayloadFormat.ARROW, pa.__version__)
        return formats

    def load_data(self, binary_data, mimetype, try_hook=True):

        if self._custom_hooks.get(CustomHooks.READ_INPUT_DATA) and try_hook:
            try:
                data = self._custom_hooks[CustomHooks.READ_INPUT_DATA](binary_data)
            except Exception as exc:
                raise type(exc)(
                    "Model read_data hook failed to read input data: {} {}".format(binary_data, exc)
                ).with_traceback(sys.exc_info()[2]) from None
        else:
            data = StructuredInputReadUtils.read_structured_input_data_as_df(
                binary_data,
                mimetype,
            )

        return data

    def preprocess(self, model=None, **kwargs):
        """
        Preprocess data and then pass on to predict method.
        Loads data, either with read hook or built-in method, and applies transform hook if present

        Parameters
        ----------
        model: Any
            The model
        Returns
        -------
        pd.DataFrame
        """
        input_binary_data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        data = self.load_data(input_binary_data, kwargs.get(StructuredDtoKeys.MIMETYPE))

        if self._custom_hooks.get(CustomHooks.TRANSFORM):
            try:
                output_data = self._custom_hooks[CustomHooks.TRANSFORM](data, model)

            except Exception as exc:
                raise type(exc)(
                    "Model transform hook failed to transform dataset: {}".format(exc)
                ).with_traceback(sys.exc_info()[2]) from None

            self._validate_data(output_data, CustomHooks.TRANSFORM)

        else:
            output_data = data

        return output_data

    def transform(self, model=None, **kwargs):
        """
        Standalone transform method, only used for custom transforms.

        Parameters
        ----------
        model: Any
            The model
        Returns
        -------
        pd.DataFrame
        """
        input_binary_data = kwargs.get(StructuredDtoKeys.BINARY_DATA)
        target_binary_data = kwargs.get(StructuredDtoKeys.TARGET_BINARY_DATA)

        data = self.load_data(input_binary_data, kwargs.get(StructuredDtoKeys.MIMETYPE))
        target_data = None

        if target_binary_data:
            target_data = self.load_data(
                target_binary_data, kwargs.get(StructuredDtoKeys.TARGET_MIMETYPE), try_hook=False
            )

        if self._custom_hooks.get(CustomHooks.TRANSFORM):
            try:
                transform_hook = self._custom_hooks[CustomHooks.TRANSFORM]
                transform_params = signature(transform_hook).parameters
                if len(transform_params) == 3:
                    transform_out = transform_hook(data, model, target_data)
                elif len(transform_params) == 2:
                    transform_out = transform_hook(data, model)
                else:
                    raise ValueError(
                        "Transform hook must take 2 or 3 arguments; "
                        "hook provided takes {}".format(len(transform_params))
                    )
                if type(transform_out) == tuple:
                    output_data, output_target = transform_out
                else:
                    output_data = transform_out
                    output_target = target_data

            except Exception as exc:
                raise type(exc)(
                    "Model transform hook failed to transform dataset: {}".format(exc)
                ).with_traceback(sys.exc_info()[2]) from None
            self._validate_data(output_data, CustomHooks.TRANSFORM)
            self._validate_transform_rows(output_data, data)
            if output_target is not None:
                self._validate_transform_rows(output_target, target_data)
            return output_data, output_target
        elif "sklearn" in str(type(model)):
            # we don't touch y if user doesn't pass a hook
            return self._apply_sklearn_transformer(data, model), target_data
        else:
            raise ValueError(
                "Transform hook must be implemented for custom transforms, "
                "for non-sklearn transformer."
            )

    def has_read_input_data_hook(self):
        return self._custom_hooks.get(CustomHooks.READ_INPUT_DATA) is not None

    def predict(self, model=None, **kwargs):
        """
        Makes predictions against the model using the custom predict
        method and returns a pandas DataFrame
        If the model is a regression model, the DataFrame will have a single column "Predictions"
        If the model is a classification model, the DataFrame will have a column for each class label
            with their respective probabilities. Positive/negative class labels will be passed in kwargs under
            positive_class_label/negative_class_label keywords.
        Parameters
        ----------
        model: Any
            The model
        kwargs
        Returns
        -------
        pd.DataFrame
        """
        data = self.preprocess(model, **kwargs)

        positive_class_label = kwargs.get(POSITIVE_CLASS_LABEL_ARG_KEYWORD)
        negative_class_label = kwargs.get(NEGATIVE_CLASS_LABEL_ARG_KEYWORD)
        class_labels = (
            [str(label) for label in kwargs.get(CLASS_LABELS_ARG_KEYWORD)]
            if kwargs.get(CLASS_LABELS_ARG_KEYWORD)
            else None
        )
        if positive_class_label is not None and negative_class_label is not None:
            class_labels = [negative_class_label, positive_class_label]

        if self._custom_hooks.get(CustomHooks.SCORE):
            try:
                # noinspection PyCallingNonCallable
                predictions = self._custom_hooks.get(CustomHooks.SCORE)(data, model, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    "Model score hook failed to make predictions. Exception: {}".format(exc)
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

        self._validate_predictions(predictions, class_labels)

        return predictions

    @staticmethod
    def _validate_unstructured_predictions(unstructured_response):
        # response can be either str/bytes or tuple
        # normalize to tuple
        single_input_value = False
        if not isinstance(unstructured_response, tuple):
            single_input_value = True
            unstructured_response = (unstructured_response, None)

        if len(unstructured_response) != 2:
            raise ValueError(
                "In unstructured mode predictions of type tuple must have length = 2, but received length = {}".format(
                    len(unstructured_response)
                )
            )

        data, kwargs = unstructured_response

        if not isinstance(data, (bytes, str, type(None))) or not isinstance(
            kwargs, (dict, type(None))
        ):
            if single_input_value:
                error_msg = "In unstructured mode single return value can be of type str/bytes, but received {}".format(
                    type(data)
                )
            else:
                error_msg = "In unstructured mode tuple return value must be of type (str/bytes, dict) but received ({}, {})".format(
                    type(data), type(kwargs)
                )
            raise ValueError(error_msg)

    def predict_unstructured(self, model, data, **kwargs):
        predictions = self._custom_hooks.get(CustomHooks.SCORE_UNSTRUCTURED)(model, data, **kwargs)
        PythonModelAdapter._validate_unstructured_predictions(predictions)
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
        import sklearn

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
            if y is not None:
                marked_object.fit(X, y)
            else:
                marked_object.fit(X, y=None)
            with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
                pickle.dump(marked_object, fp)
            return True
        return False

    def fit(self, X, y, output_dir, class_order=None, row_weights=None):
        with reroute_stdout_to_stderr():
            if self._custom_hooks.get(CustomHooks.FIT):
                self._custom_hooks[CustomHooks.FIT](
                    X=X,
                    y=y,
                    output_dir=output_dir,
                    class_order=class_order,
                    row_weights=row_weights,
                )
            elif self._drum_autofit_internal(X, y, output_dir):
                return
            else:
                hooks = [
                    "{}: {}".format(hook, fn is not None) for hook, fn in self._custom_hooks.items()
                ]
                raise DrumCommonException(
                    "\n\n{}\n"
                    "\nfit() method must be implemented in a file named 'custom.py' in the provided code_dir: '{}' \n"
                    "Here is a list of files in this dir. {}\n"
                    "Here are the hooks your custom.py file has: {}".format(
                        RUNNING_LANG_MSG, self._model_dir, os.listdir(self._model_dir)[:100], hooks
                    )
                )
