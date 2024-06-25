import glob
import math
import os
import psutil
import sys

from py4j.java_gateway import JavaGateway
from py4j.java_gateway import (
    GatewayParameters,
    CallbackServerParameters,
    launch_gateway,
)
from py4j.protocol import Py4JJavaError

from mlpiper.common.mlpiper_exception import MLPiperException
from mlpiper.common.byte_conv import ByteConv
from mlpiper.pipeline.component_runner.component_runner import ComponentRunner


class JavaConnectedComponentRunner(ComponentRunner):
    JAVA_PROGRAM = "java"
    SLEEP_INTERVAL_AFTER_RUNNING_JVM = 2
    ENTRY_POINT_CLASS = "com.mlpiper.ComponentEntryPoint"

    def __init__(self, ml_engine, dag_node, mlpiper_jar):
        super(JavaConnectedComponentRunner, self).__init__(ml_engine, dag_node)

        if not mlpiper_jar or not os.path.exists(mlpiper_jar):
            raise Exception("mlpiper_jar does not exists: {}".format(mlpiper_jar))
        self._mlpiper_jar = mlpiper_jar

    def run(self, parent_data_objs):

        # Run the java py4j entry point
        comp_dir = self._dag_node.comp_root_path()
        self._logger.info("comp_dir: {}".format(comp_dir))

        jar_files = glob.glob(os.path.join(comp_dir, "*.jar"))
        self._logger.info("Java classpath files: {}".format(jar_files))
        component_class = self._dag_node.comp_class()

        java_jars = [self._mlpiper_jar] + jar_files
        class_path = ":".join(java_jars)
        java_gateway = None
        all_ok = False

        try:
            total_phys_mem_size_mb = ByteConv.from_bytes(
                psutil.virtual_memory().total
            ).mbytes
            jvm_heap_size_option = "-Xmx{}m".format(
                int(math.ceil(total_phys_mem_size_mb))
            )

            java_opts = [jvm_heap_size_option]
            self._logger.info("JVM options: {}".format(java_opts))

            # Note: the jarpath is set to be the path to the mlpiper jar since the
            # launch_gateway code is checking for the existence of the jar. The py4j
            # jar is packed inside the mlpiper jar.
            java_port = launch_gateway(
                port=0,
                javaopts=java_opts,
                die_on_exit=True,
                jarpath=self._mlpiper_jar,
                classpath=class_path,
                redirect_stdout=sys.stdout,
                redirect_stderr=sys.stderr,
            )

            java_gateway = JavaGateway(
                gateway_parameters=GatewayParameters(port=java_port),
                callback_server_parameters=CallbackServerParameters(port=0),
            )

            python_port = java_gateway.get_callback_server().get_listening_port()
            self._logger.debug("Python port: {}".format(python_port))

            java_gateway.java_gateway_server.resetCallbackClient(
                java_gateway.java_gateway_server.getCallbackClient().getAddress(),
                python_port,
            )

            entry_point = java_gateway.jvm.com.mlpiper.ComponentEntryPoint(
                component_class
            )
            component_via_py4j = entry_point.getComponent()

            # Configure
            m = java_gateway.jvm.java.util.HashMap()
            for key in self._params.keys():
                # py4j does not handle nested structures. So the configs which is a dict
                # will not be passed to the java layer now.
                if isinstance(self._params[key], dict):
                    continue
                m[key] = self._params[key]

            component_via_py4j.configure(m)

            # Materialized
            parents_edges = java_gateway.jvm.java.util.ArrayList()
            for obj in parent_data_objs:
                parents_edges.append(obj)
                self._logger.info("Parent obj: {} type {}".format(obj, type(obj)))
            self._logger.info("Parent objs: {}".format(parents_edges))

            py4j_out_objs = component_via_py4j.materialize(parents_edges)

            self._logger.debug(type(py4j_out_objs))
            self._logger.debug(len(py4j_out_objs))

            python_out_objs = []
            for obj in py4j_out_objs:
                self._logger.debug("Obj:")
                self._logger.debug(obj)
                python_out_objs.append(obj)
            self._logger.info("Done running of materialize and getting output objects")
            all_ok = True
        except Py4JJavaError as e:
            self._logger.error("Error in java code: {}".format(e))
            raise MLPiperException(str(e))
        except Exception as e:
            self._logger.error("General error: {}".format(e))
            raise MLPiperException(str(e))
        finally:
            self._logger.info("In finally block: all_ok {}".format(all_ok))
            if java_gateway:
                java_gateway.close_callback_server()
                java_gateway.shutdown()

        return python_out_objs
