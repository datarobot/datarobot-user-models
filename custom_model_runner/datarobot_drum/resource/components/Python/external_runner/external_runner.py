from mlpiper.components.connectable_component import ConnectableComponent
from mlpiper.pipeline.executor import Executor
from mlpiper.pipeline.executor_config import ExecutorConfig
from mlpiper.pipeline import json_fields
from mlpiper.cli.mlpiper_runner import MLPiperRunner

from jinja2 import Template
import json
import os
import pickle
import pandas as pd
import tempfile


class ShmemFormat(object):
    PICKLE = "pickle"
    CSV = "csv"


class ShmemUtils(object):
    @staticmethod
    def create_in_mem_file(shmem_format):
        fd = None
        if shmem_format == ShmemFormat.PICKLE:
            fd = tempfile.NamedTemporaryFile(mode="w+b")
        elif shmem_format == ShmemFormat.CSV:
            fd = tempfile.NamedTemporaryFile(mode="w")
        return fd

    @staticmethod
    def create_out_mem_file(shmem_format):
        fd = None
        if shmem_format == ShmemFormat.PICKLE:
            fd = tempfile.NamedTemporaryFile(mode="r+b")
        elif shmem_format == ShmemFormat.CSV:
            fd = tempfile.NamedTemporaryFile(mode="r")
        return fd

    @staticmethod
    def write_in_mem_file(shmem_format, df, fd):
        fd.truncate(0)
        fd.seek(0)
        if shmem_format == ShmemFormat.PICKLE:
            pickle.dump(df, fd)
        elif shmem_format == ShmemFormat.CSV:
            df.to_csv(fd, index=False)
        fd.flush()

    @staticmethod
    def read_out_mem_file(shmem_format, fd):
        fd.seek(0)
        if shmem_format == ShmemFormat.PICKLE:
            df = pickle.load(fd)
        elif shmem_format == ShmemFormat.CSV:
            df = pd.read_csv(fd)
        return df


class ExternalRunner(ConnectableComponent):
    def __init__(self, engine):
        super(ExternalRunner, self).__init__(engine)
        self._pipeline_executor = None

        self._in_mmap_file = None
        self._out_mmap_file = None

        self._shmem_format = None

    def __del__(self):
        self._in_mmap_file.close()
        self._out_mmap_file.close()

    def _fix_pipeline(self, pipeline_str):
        pipeline_json = json.loads(pipeline_str)

        if json_fields.PIPELINE_SYSTEM_CONFIG_FIELD not in pipeline_json:
            system_config = {}
            pipeline_json[json_fields.PIPELINE_SYSTEM_CONFIG_FIELD] = system_config
        return json.dumps(pipeline_json)

    def configure(self, params):
        super(ExternalRunner, self).configure(params)

        self._shmem_format = self._params.get("shmem_format")
        if self._shmem_format is None:
            self._shmem_format = ShmemFormat.PICKLE

        self._in_mmap_file = ShmemUtils.create_in_mem_file(self._shmem_format)
        self._out_mmap_file = ShmemUtils.create_out_mem_file(self._shmem_format)

        replace_data = {
            "input_filename": self._in_mmap_file.name,
            "output_filename": '"{}"'.format(self._out_mmap_file.name),
        }

        pipeline_str = self._params.get("pipeline")
        pipeline_str = Template(pipeline_str).render(replace_data)
        pipeline_str = self._fix_pipeline(pipeline_str)

        comp_repo = self._params.get("repo")

        config = ExecutorConfig(
            pipeline=pipeline_str,
            pipeline_file=None,
            run_locally=True,
            comp_root_path=comp_repo,
            mlpiper_jar=os.path.join(MLPiperRunner.SCRIPT_DIR, "..", "jars", "mlpiper.jar"),
            spark_jars=None,
        )

        self._pipeline_executor = Executor(config)
        self._pipeline_executor.init_pipeline()

    def _get_out_df(self):
        return ShmemUtils.read_out_mem_file(self._shmem_format, self._out_mmap_file)

    def _clean_out_mem(self):
        pass

    def _set_in_df(self, in_df):
        ShmemUtils.write_in_mem_file(self._shmem_format, in_df, self._in_mmap_file)

    def _run_pipeline(self):
        self._pipeline_executor.run_pipeline(cleanup=False)

    def _cleanup_pipeline(self):
        self._pipeline_executor.cleanup_pipeline()

    def _materialize(self, parent_data_objs, user_data):
        self._pipeline_executor.go()
        return []
