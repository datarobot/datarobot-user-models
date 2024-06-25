import abc
from mlpiper.components.spark_session_component import SparkSessionComponent


class SparkDataComponent(SparkSessionComponent):
    def __init__(self, ml_engine):
        super(SparkDataComponent, self).__init__(ml_engine)

    def _materialize(self, spark, parent_data_objs, user_data):
        df = self._dataframe(spark, user_data)
        df_list = df if type(df) is list else [df]
        self._logger.debug(
            "Data component '{}' returns: {}".format(self.name(), df_list)
        )
        return df_list  # Used by child connectable component

    def _post_validation(self, df):
        self._ml_engine.set_dataframe(df)  # Used by Spark ml pipeline
        return df

    @abc.abstractmethod
    def _dataframe(self, spark, user_data):
        """
        Supposed to return spark data-frame
        """
        pass
