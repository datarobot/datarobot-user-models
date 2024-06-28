import abc

from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components.spark_session_component import SparkSessionComponent


class SparkStageComponent(SparkSessionComponent):
    def __init__(self, ml_engine):
        super(SparkStageComponent, self).__init__(ml_engine)
        global Estimator, Transformer

        try:
            from pyspark.ml.base import Estimator, Transformer
        except ImportError:
            raise MLPiperException(
                "'pyspark' python package is missing! "
                "You may install it using: 'pip install pyspark'!"
            )

    def _materialize(self, spark, parent_data_objs, user_data):
        return self._stage(self._ml_engine.session, self._ml_engine.user_data)

    def _validate_output(self, stage):
        if stage:
            if not issubclass(stage.__class__, Transformer) and not issubclass(
                stage.__class__, Estimator
            ):
                raise MLPiperException(
                    "Invalid returned stage type! Expecting for 'Transformer' or 'Estimator'! "
                    "name: {}, type: {}".format(self.name(), type(stage))
                )

    def _post_validation(self, stage):
        self._ml_engine.add_stage(stage)
        return None

    def _set_interim_directory(self, tmp_path):
        self._ml_engine.set_interim_directory(tmp_path)

    def _set_output_model_path(self, path):
        self._ml_engine.set_output_model_path(path)

    @abc.abstractmethod
    def _stage(self, spark, user_data):
        """
        This method is supposed to return a single spark ml pipeline stage
        """
        pass
