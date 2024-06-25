import abc

from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components.component import Component


class SparkSessionComponent(Component):
    def __init__(self, ml_engine):
        super(SparkSessionComponent, self).__init__(ml_engine)

    def materialize(self, parent_data_objs):
        outs = self._materialize(
            self._ml_engine.session, parent_data_objs, self._ml_engine.user_data
        )
        self._validate_output(outs)
        if outs:
            outs = self._post_validation(outs)
        return outs

    def _validate_output(self, dfs):
        if dfs:
            if type(dfs) is not list:
                raise MLPiperException(
                    "Invalid non-list output! Expecting a list of 'pyspark.sql.DataFrame'! "
                    "name: {}, type: {}".format(self.name(), type(dfs))
                )

    def _post_validation(self, dfs):
        """
        This method is supposed to return a revised list of data-frame object(s) or None.
        By default is returns the original list of data-frame object(s)
        (Overridable)
        """
        return dfs

    def _set_interim_directory(self, tmp_path):
        self._ml_engine.set_interim_directory(tmp_path)

    def _get_input_model(self, path):
        return self._ml_engine.get_input_model(model_path=path)

    @abc.abstractmethod
    def _materialize(self, spark, parent_data_objs, user_data):

        """
        This abstract method is supposed to return a list of spark data-frame object(s)
        """
        pass
