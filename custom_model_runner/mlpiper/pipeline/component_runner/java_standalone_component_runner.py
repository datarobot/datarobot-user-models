import glob
import os
import subprocess
import sys

from mlpiper.pipeline.component_runner.standalone_component_runner import (
    StandaloneComponentRunner,
)
from mlpiper.pipeline.pipeline_utils import assemble_cmdline_from_args


class JavaStandaloneComponentRunner(StandaloneComponentRunner):
    JAVA_PROGRAM = "java"

    def __init__(self, ml_engine, dag_node):
        super(JavaStandaloneComponentRunner, self).__init__(ml_engine, dag_node)
        self._dag_node = dag_node

    # TODO: move this to parent class
    def _run_external_process(self, cmd, workdir):
        self.info("CMD: {}".format(cmd))

        os.chdir(workdir)
        # Save env variables should be passed
        self._logger.info("================== External code start ==================")
        sys.stdout.flush()
        p = subprocess.Popen(cmd)
        p.wait()
        self._logger.info(
            "================= External code done: ret: {} =================".format(
                p.returncode
            )
        )

        sys.stdout.flush()
        if p.returncode != 0:
            self._logger.info(
                "Connector: got external program exit code: {}".format(p.returncode)
            )
        return p.returncode

    def run(self, parent_data_objs):
        self._logger.info("Materialize for Java standalone")

        comp_dir = self._dag_node.comp_root_path()
        print("comp_dir: {}".format(comp_dir))

        jar_files = glob.glob(os.path.join(comp_dir, "*.jar"))
        self._logger.info("Java classpath files: {}".format(jar_files))

        class_path = ":".join(jar_files)
        class_name = self._dag_node.comp_class()
        cmd = []
        cmd.extend(
            [JavaStandaloneComponentRunner.JAVA_PROGRAM, "-cp", class_path, class_name]
        )

        component_cmdline = assemble_cmdline_from_args(self._params)
        self._logger.debug("cmdline: {}".format(component_cmdline))

        cmd.extend(component_cmdline)
        ret_code = self._run_external_process(cmd, comp_dir)

        if ret_code != 0:
            raise Exception(
                "Java component exited with exit status {}".format(ret_code)
            )
