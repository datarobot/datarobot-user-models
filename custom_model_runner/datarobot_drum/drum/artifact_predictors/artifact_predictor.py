from abc import ABC, abstractmethod
import logging

from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.common import (
    LOGGER_NAME_PREFIX,
    POSITIVE_CLASS_LABEL_ARG_KEYWORD,
    NEGATIVE_CLASS_LABEL_ARG_KEYWORD,
    CLASS_LABELS_ARG_KEYWORD,
    TARGET_TYPE_ARG_KEYWORD,
    TargetType,
)


class ArtifactPredictor(ABC):
    def __init__(self, name, suffix):
        self._name = name
        self._artifact_extension = suffix
        self.positive_class_label = None
        self.negative_class_label = None
        self.class_labels = None
        self.target_type = None
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
        self.positive_class_label = kwargs.get(POSITIVE_CLASS_LABEL_ARG_KEYWORD)
        self.negative_class_label = kwargs.get(NEGATIVE_CLASS_LABEL_ARG_KEYWORD)
        self.class_labels = kwargs.get(CLASS_LABELS_ARG_KEYWORD)
        self.target_type = kwargs[TARGET_TYPE_ARG_KEYWORD]
        if self.target_type == TargetType.BINARY and None in [
            self.positive_class_label,
            self.negative_class_label,
        ]:
            raise DrumCommonException(
                "For `{}` target type both class labels must be provided. Found: {}, {}".format(
                    self.target_type.value, self.positive_class_label, self.negative_class_label
                )
            )
        if self.target_type == TargetType.MULTICLASS and not self.class_labels:
            raise DrumCommonException(
                "For `{}` target, the model's class labels must be provided. found: {}".format(
                    self.target_type, self.class_labels
                )
            )
        if self.target_type == TargetType.BINARY:
            self.class_labels = [self.negative_class_label, self.positive_class_label]
