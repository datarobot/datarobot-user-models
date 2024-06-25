WSGI_ENTRY_SCRIPT = """
import logging

from mlpiper.ml_engine.rest_model_serving_engine import RestModelServingEngine
from {module} import {cls}
from {restful_comp_module} import {restful_comp_cls}


logging.basicConfig(format='{log_format}')
logging.getLogger('{root_logger_name}').setLevel({log_level})

comp = {restful_comp_cls}(None)
comp.configure({params})

{cls}.uwsgi_entry_point(
    comp,
    '{pipeline_name}',
    '{model_path}',
    '{deputy_id}',
    '{stats_path_filename}',
    within_uwsgi_context={within_uwsgi_context},
    standalone={standalone}
)

application = {cls}._application
"""
