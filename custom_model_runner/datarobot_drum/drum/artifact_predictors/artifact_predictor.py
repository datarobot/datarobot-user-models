from abc import ABC, abstractmethod
import logging

from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
)


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
