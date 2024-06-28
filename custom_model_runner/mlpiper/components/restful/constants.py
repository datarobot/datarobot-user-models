from mlpiper.common.os_util import service_installed


class SharedConstants:
    # Keys for shared configuration dict
    TARGET_PATH_KEY = "target_path"
    SOCK_FILENAME_KEY = "sock_filename"
    STATS_SOCK_FILENAME_KEY = "stats_sock_filename"
    REST_COMP_ROOT_PATH = "rest_comp_root_path"
    STANDALONE = "standalone"
    STATS_PATH_FILENAME_KEY = "stats_path_filename"


class ComponentConstants:
    CONFIGURE_CALLBACK_FUNC_NAME = "configure_callback"
    LOAD_MODEL_CALLBACK_FUNC_NAME = "load_model_callback"
    CLEANUP_CALLBACK_FUNC_NAME = "cleanup_callback"
    POST_FORK_CALLBACK_FUNC_NAME = "post_fork_callback"

    TMP_RESTFUL_ROOT = "/tmp"
    TMP_RESTFUL_DIR_PREFIX = "restful_comp_"
    INPUT_MODEL_TAG_NAME = "input_model_path"

    # *** RESTful component keys ***

    PORT_KEY = "port"

    HOST_KEY = "host"
    DEFAULT_HOST = "localhost"

    LOG_FORMAT_KEY = "log_format"
    DEFAULT_LOG_FORMAT = "%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s"

    LOG_LEVEL_KEY = "log_level"
    DEFAULT_LOG_LEVEL = "info"

    # Disable 'uwsgi' requests logging
    UWSGI_DISABLE_LOGGING_KEY = "uwsgi_disable_logging"
    DEFAULT_UWSGI_DISABLE_LOGGING = True
    UWSGI_MAX_WORKERS_KEY = "uwsgi_max_workers"
    UWSGI_CHEAPER_RSS_LIMIT_SOFT_KEY = "uwsgi_cheaper_rss_limit_soft"
    UWSGI_CHEAPER_RSS_LIMIT_HARD_KEY = "uwsgi_cheaper_rss_limit_hard"
    UWSGI_MEMORY_REPORT_KEY = "uwsgi_memory_report"
    SINGLE_UWSGI_WORKER_KEY = "single_uwsgi_worker"

    METRICS_KEY = "metrics"
    METRIC_TEMPLATE = "metric = name={},type=counter,initial_value=0,oid=100.{}"

    # Specify the reporting interval as well as the time period that stats metrics are referred to
    STATS_REPORTING_INTERVAL_SEC = "stats_reporting_interval_sec"
    DEFAULT_STATS_REPORTING_INTERVAL_SEC = 10

    # The dry run key, is for internal use only. It design to skip the execution of uwsgi & nginx
    # applications. It is used for debugging, when someone wants to analyse all the configurations
    # before actually starting up given processes
    DRY_RUN_KEY = "__dry_run__"
    DEFAULT_DRY_RUN = False


class UwsgiConstants:
    DEV_AGAINST_VERSION = "2.0.17.1"

    START_CMD = "uwsgi --ini {filepath}"
    STOP_CMD = "uwsgi --stop {pid_filepath}"
    VER_CMD = "uwsgi --version"

    DAEMONIZE = False
    INI_FILENAME = "uwsgi.ini"
    PID_FILENAME = "uwsgi.pid"
    ENTRY_POINT_SCRIPT_NAME = "uwsgi_entry_script.py"
    SOCK_FILENAME = "restful_mlapp.sock"
    STATS_SOCK_FILENAME = "stats.restful_mlapp.sock"

    MONITOR_THREAD_KEY = "monitor_th"
    MONITOR_ERROR_KEY = "error"

    MODEL_RELOAD_SIGNAL_NUM = 13

    # *** Keys for 'uwsgi' configuration dict ***

    RESTFUL_COMP_MODULE_KEY = "restful_comp_module"
    RESTFUL_COMP_CLS_KEY = "restful_comp_cls"
    PARAMS_KEY = "params"
    PIPELINE_NAME_KEY = "pipeline_name"
    MODEL_PATH_KEY = "model_path"
    DEPUTY_ID_KEY = "deputy_id"
    WORKER_ID = "worker_id"
    LOGGING_UDP_SOCKET = "logging_udp_socket"


class RestfulConstants:
    STATS_ROUTE = "statsinternal"
    STATS_SYSTEM_INFO = "sys_info"
    STATS_WID = "wid"
    STATS_WUUID = "wuuid"
    STATS_UUID = "uuid2"
    STATS_USER = "user"
    STATS_SYSTEM = "system"
    STATS_SYSTEM_ERROR = "system_error"
    STATS_AGGREGATE_FLAG = "AGGREGATE_STATS"


class NginxConstants:
    DEV_AGAINST_VERSION = "nginx/1.10.3"

    START_CMD = "service nginx start" if service_installed("dbus") else "nginx"
    STOP_CMD = "service nginx stop" if service_installed("dbus") else "nginx -s quit"
    VER_CMD = "nginx -v"

    SERVER_CONF_FILENAME = "mlpiper.pipeline.restful"

    NGINX_ROOT = "/etc/nginx"
    SERVER_CONF_DIR_DEBIAN = NGINX_ROOT + "/sites-available"
    SERVER_CONF_DIR_REDHAT = NGINX_ROOT + "/conf.d"

    NGINX_ROOT_MACOS = "/usr/local/etc/nginx"
    SERVER_CONF_DIR_MACOS = NGINX_ROOT_MACOS + "/servers"

    SERVER_ENABLED_DIR = NGINX_ROOT + "/sites-enabled"

    DISABLE_ACCESS_LOG_KEY = "disable_access_log"
    ACCESS_LOG_OFF_CONFIG = "access_log   off;"


class StatsConstants:
    REQS_PER_WINDOW_TIME_GRAPH_TITLE = "Total Requests / {}sec"

    ACC_REQS_TABLE_NAME = "Accumulated REST requests"
    ACC_REQS_NUM_REQS_COL_NAME = "Num Requests"
    ACC_REQS_STATUS_COL_NAME = "Status"
    ACC_REQS_LAST_ROW_NAME = "Total"

    AVG_RESP_TIME_TABLE_NAME = "Average response time"
    AVG_RESP_TIME_COL_NAME = "Time [us]"
