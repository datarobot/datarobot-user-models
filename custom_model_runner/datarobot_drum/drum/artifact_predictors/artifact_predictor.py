from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd
import logging
import sys

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
)
from datarobot_drum.drum.exceptions import DrumCommonException


class ArtifactPredictor(ABC):
    def __init__(self, name, suffix):
        self._name = name
        self._artifact_extension = suffix
        self.positive_class_label = None
        self.negative_class_label = None
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)

    @property
    def name(self):
        return self._name

    @property
    def artifact_extension(self):
        return self._artifact_extension

    def is_artifact_supported(self, artifact_path):
        if artifact_path.endswith(self._artifact_extension):
            return True
        else:
            return False

    @abstractmethod
    def framework_requirements(self):
        """ Return a list of the framework python requirements"""
        pass

    @abstractmethod
    def is_framework_present(self):
        """ Check if the framework can be loaded """
        pass

    @abstractmethod
    def can_load_artifact(self, artifact_path):
        """ Check if the model artifact can be loaded """
        pass

    @abstractmethod
    def load_model_from_artifact(self, artifact_path):
        """ Load the model artifact to a model object"""
        pass

    @abstractmethod
    def can_use_model(self, model):
        """ Given a model object, can this predictor use the given model"""
        pass

    @abstractmethod
    def predict(self, data, model, **kwargs):
        """
        Run prediction on this model
        """
        self.positive_class_label = kwargs.get(POSITIVE_CLASS_LABEL_ARG_KEYWORD, None)
        self.negative_class_label = kwargs.get(NEGATIVE_CLASS_LABEL_ARG_KEYWORD, None)


class SKLearnPredictor(ArtifactPredictor):
    def __init__(self):
        super(SKLearnPredictor, self).__init__(
            SupportedFrameworks.SKLEARN, PythonArtifacts.PKL_EXTENSION
        )

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.SKLEARN]

    def is_framework_present(self):
        try:
            from sklearn.base import BaseEstimator

            return True
        except ImportError as e:
            return False

    def can_load_artifact(self, artifact_path):
        return self.is_artifact_supported(artifact_path)

    def load_model_from_artifact(self, artifact_path):
        with open(artifact_path, "rb") as picklefile:
            try:
                model = pickle.load(picklefile, encoding="latin1")
            except TypeError:
                model = pickle.load(picklefile)
            return model

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            return True
        else:
            return False

    def predict(self, data, model, **kwargs):
        # checking if positive/negative class labels were provided
        # done in the base class
        super(SKLearnPredictor, self).predict(data, model, **kwargs)

        def _determine_positive_class_index(pos_label, neg_label):
            """Find index of positive class label to interpret predict_proba output"""

            if not hasattr(model, "classes_"):
                self._logger.warning(
                    "We were not able to verify you were using the right class labels because your estimator doesn't have a classes_ attribute"
                )
                return 1
            labels = [str(label) for label in model.classes_]
            if not all(x in labels for x in [pos_label, neg_label]):
                error_message = "Wrong class labels. Use class labels detected by sklearn model: {}".format(
                    labels
                )
                raise DrumCommonException(error_message)

            return labels.index(pos_label)

        if self.positive_class_label is not None and self.negative_class_label is not None:
            predictions = model.predict_proba(data)
            positive_label_index = _determine_positive_class_index(
                self.positive_class_label, self.negative_class_label
            )
            negative_label_index = 1 - positive_label_index
            predictions = [
                [prediction[positive_label_index], prediction[negative_label_index]]
                for prediction in predictions
            ]
            predictions = pd.DataFrame(
                predictions, columns=[self.positive_class_label, self.negative_class_label]
            )
        else:
            predictions = pd.DataFrame(
                [float(prediction) for prediction in model.predict(data)],
                columns=[REGRESSION_PRED_COLUMN],
            )

        return predictions


class KerasPredictor(ArtifactPredictor):
    def __init__(self):
        super(KerasPredictor, self).__init__(
            SupportedFrameworks.KERAS, PythonArtifacts.KERAS_EXTENSION
        )
        self._model = None

    def is_framework_present(self):
        try:
            import keras
            from keras.models import load_model

            return True
        except ImportError:
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.KERAS]

    def can_load_artifact(self, artifact_path):
        if not self.is_artifact_supported(artifact_path):
            return False

        try:
            import keras
            from keras.models import load_model

            return True
        except ImportError:
            return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        try:
            import keras
            from sklearn.pipeline import Pipeline

            if isinstance(model, Pipeline):
                # check the final estimator in the pipeline is Keras
                if isinstance(model[-1], keras.Model):
                    return True
            elif isinstance(model, keras.Model):
                return True
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False
        return False

    def load_model_from_artifact(self, artifact_path):
        from keras.models import load_model

        self._model = load_model(artifact_path)
        self._model._make_predict_function()
        return self._model

    def predict(self, data, model, **kwargs):
        # checking if positive/negative class labels were provided
        # done in the base class
        super(KerasPredictor, self).predict(data, model, **kwargs)
        predictions = model.predict(data)
        if self.positive_class_label is not None and self.negative_class_label is not None:
            if predictions.shape[1] == 1:
                predictions = pd.DataFrame(predictions, columns=[self.positive_class_label])
                predictions[self.negative_class_label] = 1 - predictions[self.positive_class_label]
            else:
                predictions = pd.DataFrame(
                    predictions, columns=[self.negative_class_label, self.positive_class_label]
                )
        else:
            predictions = pd.DataFrame(predictions, columns=[REGRESSION_PRED_COLUMN])

        return predictions


class PyTorchPredictor(ArtifactPredictor):
    def __init__(self):
        super(PyTorchPredictor, self).__init__(
            SupportedFrameworks.TORCH, PythonArtifacts.TORCH_EXTENSION
        )

    def is_framework_present(self):
        try:
            import torch
            import torch.nn as nn
            from torch.autograd import Variable

            return True
        except ImportError as e:
            self._logger.debug("Got error in imports: {}".format(e))
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.TORCH]

    def can_load_artifact(self, artifact_path):
        if self.is_artifact_supported(artifact_path) and self.is_framework_present():
            return True
        return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False
        try:
            import torch
            import torch.nn as nn

            if isinstance(model, nn.Module):
                return True
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False

    def load_model_from_artifact(self, artifact_path):
        self._logger.debug("sys_path: {}".format(sys.path))
        import torch

        model = torch.load(artifact_path)
        model.eval()
        return model

    def predict(self, data, model, **kwargs):
        import torch
        from torch.autograd import Variable

        # checking if positive/negative class labels were provided
        # done in the base class
        super(PyTorchPredictor, self).predict(data, model, **kwargs)

        data = Variable(torch.from_numpy(data.values).type(torch.FloatTensor))
        with torch.no_grad():
            predictions = model(data).cpu().data.numpy()
        if self.positive_class_label is not None and self.negative_class_label is not None:
            if predictions.shape[1] == 1:
                predictions = pd.DataFrame(predictions, columns=[self.positive_class_label])
                predictions[self.negative_class_label] = 1 - predictions[self.positive_class_label]
            else:
                predictions = pd.DataFrame(
                    predictions, columns=[self.negative_class_label, self.positive_class_label]
                )
        else:
            predictions = pd.DataFrame(predictions, columns=[REGRESSION_PRED_COLUMN])
        return predictions


class XGBoostPredictor(ArtifactPredictor):
    """
    This Predictor supports both XGBoost native & sklearn api wrapper as well
    """

    def __init__(self):
        super(XGBoostPredictor, self).__init__(
            SupportedFrameworks.XGBOOST, PythonArtifacts.PKL_EXTENSION
        )

    def is_framework_present(self):
        try:
            import xgboost

            return True
        except ImportError as e:
            self._logger.debug("Got error in imports: {}".format(e))
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.XGBOOST]

    def can_load_artifact(self, artifact_path):
        if self.is_artifact_supported(artifact_path) and self.is_framework_present():
            return True
        return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False
        try:
            from sklearn.pipeline import Pipeline
            import xgboost

            if isinstance(model, Pipeline):
                # check the final estimator in the pipeline is XGBoost
                if isinstance(
                    model[-1], (xgboost.sklearn.XGBClassifier, xgboost.sklearn.XGBRegressor)
                ):
                    return True
            elif isinstance(model, xgboost.core.Booster):
                return True
            return False
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False

    def load_model_from_artifact(self, artifact_path):
        with open(artifact_path, "rb") as picklefile:
            try:
                model = pickle.load(picklefile, encoding="latin1")
            except TypeError:
                model = pickle.load(picklefile)
            return model

    def predict(self, data, model, **kwargs):
        # checking if positive/negative class labels were provided
        # done in the base class
        super(XGBoostPredictor, self).predict(data, model, **kwargs)

        import xgboost

        xgboost_native = False
        if isinstance(model, xgboost.core.Booster):
            xgboost_native = True
            data = xgboost.DMatrix(data)

        if None not in (self.positive_class_label, self.negative_class_label):
            if xgboost_native:
                positive_preds = model.predict(data)
                negative_preds = 1 - positive_preds
                predictions = np.concatenate(
                    (positive_preds.reshape(-1, 1), negative_preds.reshape(-1, 1)), axis=1
                )
            else:
                predictions = model.predict_proba(data)
            predictions = pd.DataFrame(
                predictions, columns=[self.negative_class_label, self.positive_class_label]
            )
        else:
            preds = model.predict(data)
            predictions = pd.DataFrame(data=preds, columns=[REGRESSION_PRED_COLUMN],)

        return predictions
