import abc

from mlpiper.components.spark_session_component import SparkSessionComponent
from mlpiper.common.mlpiper_exception import MLPiperException


class SparkContextComponent(SparkSessionComponent):
    def __init__(self, ml_engine):
        super(SparkContextComponent, self).__init__(ml_engine)
        global RDD

        try:
            from pyspark.rdd import RDD
        except ImportError:
            raise MLPiperException(
                "'pyspark' python package is missing! "
                "You may install it using: 'pip install pyspark'!"
            )

    def materialize(self, parent_data_objs):
        rdds = self._materialize(
            self._ml_engine.context, parent_data_objs, self._ml_engine.user_data
        )
        return rdds

    def _validate_output(self, rdds):
        if rdds:
            if type(rdds) is not list:
                raise MLPiperException(
                    "Invalid non-list output! Expecting for a list of RDDs!"
                )

            for rdd in rdds:
                if not issubclass(rdd.__class__, RDD):
                    raise MLPiperException(
                        "Invalid returned list of rdd types! Expecting for 'pyspark.rdd.RDD'! "
                        "name: {}, type: {}".format(self.name(), type(rdd))
                    )

    @abc.abstractmethod
    def _materialize(self, sc, parent_data_objs, user_data):
        pass
