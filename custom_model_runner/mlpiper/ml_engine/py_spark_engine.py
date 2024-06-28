from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.components.spark_pipeline_model_helper import SparkPipelineModelHelper
from mlpiper.ml_engine.ml_engine import MLEngine


class PySparkEngine(MLEngine):
    def __init__(self, pipeline, run_locally, spark_jars=None, standalone=False):
        super(PySparkEngine, self).__init__(pipeline, standalone)

        global Pipeline, SparkSession

        try:
            from pyspark.ml import Pipeline
            from pyspark.sql import SparkSession
        except ImportError:
            raise MLPiperException(
                "'pyspark' python package is missing! "
                "You may install it using: 'pip install pyspark'!"
            )

        self._run_locally = run_locally
        self._spark_session = None
        self._dataframe = None
        self._output_model_path = None
        self._stages = []
        self._init_session(spark_jars)
        self._interim_directory_path = "/tmp"

        self.set_logger(self.get_engine_logger(self.logger_name()))

    def _init_session(self, spark_jars):
        spark_builder = SparkSession.builder.appName(self.pipeline_name)
        if self._run_locally:
            spark_builder.master("local[*]")

        if spark_jars:
            spark_builder.config("spark.jars", spark_jars)

        self._spark_session = spark_builder.getOrCreate()

    def _session(self):
        if not self._spark_session:
            raise MLPiperException("Spark session is not initialized!")
        return self._spark_session

    def _context(self):
        return self.session.sparkContext

    def get_engine_logger(self, name):
        spark = self.session
        logger = spark._jvm.org.apache.log4j.Logger
        return logger.getLogger(name)

    def finalize(self):
        if self._output_model_path:
            pipeline = Pipeline(stages=self._stages)
            df = self._dataframe if type(self._dataframe) is not list else self._dataframe[0]
            model = pipeline.fit(df)

            # save model to file
            model_helper = SparkPipelineModelHelper(self._context())
            model_helper.set_interim_path_to_use_for_model(
                tmp_model_path=self._interim_directory_path
            )
            model_helper.set_logger(self._logger)
            model_helper.save_model(model, self._output_model_path)

    def cleanup(self):
        if self.session:
            self.session.stop()

    def add_stage(self, stage):
        self._stages.append(stage)

    def set_dataframe(self, dataframe):
        if self._dataframe:
            if type(self._dataframe) is list:
                raise MLPiperException(
                    "DataFrame list was already set for the given pipeline! pipeline: {}, ".format(
                        self.name()
                    )
                )
            else:
                raise MLPiperException(
                    "DataFrame was already set for the given pipeline! pipeline: {}, "
                    "existing-columns: {}, new-columns: {}".format(
                        self.name(), self._dataframe.columns, dataframe.columns
                    )
                )
        self._dataframe = dataframe

    def set_output_model_path(self, path):
        if self._output_model_path:
            raise MLPiperException(
                "Output model path was already set for the given pipeline! pipeline: {}, "
                "existing-path: {}, new-path: {}".format(self.name(), self._output_model_path, path)
            )
        self._output_model_path = path

    def get_input_model(self, model_path):
        if self.context is not None:
            # load model from file
            model_helper = SparkPipelineModelHelper(self.context)
            model_helper.set_logger(self._logger)
            model_helper.set_interim_path_to_use_for_model(
                tmp_model_path=self._interim_directory_path
            )

            return model_helper.load_model(model_path)

        else:
            self._logger.error("Spark Context is not set. Failed to get model!")

    def set_interim_directory(self, interim_path):
        """
        Method is responsible for providing interim directory access for MCenter to
        temporary save model!
        :param interim_path: Path to use as interim purpose
        :return:
        """
        self._interim_directory_path = interim_path
