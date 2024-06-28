import collections

MLPIPER_JAR_ARG = "mlpiper_jar"
SPARK_JARS_ARG = "spark_jars"
SPARK_JARS_ENV_VAR = SPARK_JARS_ARG.upper()

ExecutorConfig = collections.namedtuple(
    "ExecutorConfig",
    "pipeline pipeline_file run_locally comp_root_path {} {}".format(
        MLPIPER_JAR_ARG, SPARK_JARS_ARG
    ),
)
