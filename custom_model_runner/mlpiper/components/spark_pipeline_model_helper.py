import os
import uuid

from mlpiper.common.base import Base
from mlpiper.common.mlpiper_exception import MLPiperException


class SparkPipelineModelHelper(Base):
    """
    Class is responsible for providing abstract apis to load and save Spark ML model.
    TODO: We are using HDFS as interim fs to save file before it goes to backend. We need
          proper mechanism and nit mechanism.
    """

    def __init__(self, spark_context):
        super(SparkPipelineModelHelper, self).__init__()
        global PipelineModel

        try:
            from pyspark.ml import PipelineModel
        except ImportError:
            raise MLPiperException(
                "'pyspark' python package is missing! "
                "You may install it using: 'pip install pyspark'!"
            )
        self._spark_context = spark_context
        self._model = None

        # they needs to be provided by user
        self._hdfs_tmp_dir = None

        # this are preset variables.
        self._model_dir_name_prefix = "model_dir_"
        self._model_dir_name = "MODEL"

    def set_interim_path_to_use_for_model(self, tmp_model_path):
        """
        This has to be done by user before load or save method has been invoked.
        This path represents scratch space provided by user.

        :param tmp_model_path:
        :return:
        """
        self._hdfs_tmp_dir = tmp_model_path
        return self

    def load_model(self, model_path):
        self._logger.info("model helper model_path= {}".format(model_path))

        # going into model directory to find right directory containing metadata.
        # backend adds extra directory and need to find right model directory that has metadata

        for file in os.listdir(model_path):
            # use os's join functionality
            directory_inside_model_path = model_path + "/" + file
            for each_file in os.listdir(directory_inside_model_path):
                if "metadata" in each_file:
                    model_path = directory_inside_model_path
                    break

        if self._hdfs_tmp_dir is not None:
            local_model_path = "file://" + model_path
            self._logger.info("model file will be located at: {}".format(local_model_path))

            try:
                self._logger.info(
                    "Accessing HDFS to load model dir. Copying local file to hdfs and then load."
                )
                hadoop = self._spark_context._gateway.jvm.org.apache.hadoop

                # hdfs_tmp_dir - hdfs:///tmp/model_dir_5f5c6382-63b7-46e0-b246-45a71a4f5c29
                hdfs_tmp_dir = "hdfs://" + os.path.join(
                    self._hdfs_tmp_dir, self._model_dir_name_prefix + str(uuid.uuid4())
                )

                dst_path = hadoop.fs.Path(hdfs_tmp_dir)
                src_path = hadoop.fs.Path(local_model_path)

                self._logger.info(
                    "Accessing HDFS - moving from {} to {}".format(src_path, dst_path)
                )

                config = hadoop.conf.Configuration()
                hadoop_fs = hadoop.fs.FileSystem.get(config)
                # copying local model to hdfs
                hadoop_fs.moveFromLocalFile(src_path, dst_path)

                # loading from hdfs!
                model_obj = PipelineModel.load(hdfs_tmp_dir)

                self._logger.info("Accessing HDFS - loading model from: {}".format(hdfs_tmp_dir))
                hadoop_fs.delete(dst_path, True)

            except Exception as e:
                self._logger.warning("{}".format(e))
                self._logger.warning("trying simple loading instead of using hdfs")
                model_obj = PipelineModel.load(local_model_path)

            return model_obj
        else:
            return PipelineModel.load(model_path)

    def save_model(self, model, output_path):
        """
        The function saves the spark ML model as TAR to the provided path
        :param model: model to save
        :param output_path: output_model to get the file path
        :return:
        """
        self._logger.info("model helper: output_path = {}".format(output_path))

        if self._hdfs_tmp_dir is not None:
            file_model_dir_path = "file://" + output_path
            self._logger.info("model file is located at: {}".format(file_model_dir_path))

            try:
                self._logger.info("Accessing HDFS to copy model dir to local dir.")
                hadoop = self._spark_context._gateway.jvm.org.apache.hadoop

                model_dir_name = "{}{}".format(self._model_dir_name_prefix, uuid.uuid4())
                # temporary model location in hdfs
                hdfs_model_dir_path = "hdfs://" + os.path.join(self._hdfs_tmp_dir, model_dir_name)

                self._logger.info("model will be saved to {}".format(hdfs_model_dir_path))

                model.write().overwrite().save(hdfs_model_dir_path)

                src_path = hadoop.fs.Path(hdfs_model_dir_path)
                dst_path = hadoop.fs.Path(file_model_dir_path)

                self._logger.info(
                    "Accessing HDFS - moving from {} to {}".format(src_path, dst_path)
                )

                config = hadoop.conf.Configuration()
                hadoop_fs = hadoop.fs.FileSystem.get(config)
                hadoop_fs.moveToLocalFile(src_path, dst_path)

            except Exception as e:
                self._logger.warning("{}".format(e))
                self._logger.warning("trying simple loading instead of using hdfs")
                model.write().overwrite().save(file_model_dir_path)

        else:
            model.write().overwrite().save(output_path)
