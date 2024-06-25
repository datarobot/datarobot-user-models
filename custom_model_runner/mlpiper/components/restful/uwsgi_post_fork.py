import logging
import uuid
import uwsgi

from mlpiper.components.restful.constants import ComponentConstants


class UwsgiPostFork:
    _uwsgi_broker = None
    _configure_op = None
    _postfork_op = None
    _verbose = False

    @staticmethod
    def init(uwsgi_borker):
        UwsgiPostFork._uwsgi_broker = uwsgi_borker

        UwsgiPostFork._configure_op = getattr(
            uwsgi_borker.restful_comp(),
            ComponentConstants.CONFIGURE_CALLBACK_FUNC_NAME,
            None,
        )
        UwsgiPostFork._postfork_op = getattr(
            uwsgi_borker.restful_comp(),
            ComponentConstants.POST_FORK_CALLBACK_FUNC_NAME,
            None,
        )
        UwsgiPostFork._verbose = uwsgi_borker.w_logger.isEnabledFor(logging.DEBUG)

        # Note: it is necessary to enable the uWSGI master process to use
        # 'uwsgidecorators' module
        import uwsgidecorators
        uwsgidecorators.postfork(UwsgiPostFork.do_post_fork)

    @staticmethod
    def do_post_fork():
        if UwsgiPostFork._verbose:
            UwsgiPostFork._uwsgi_broker.w_logger.debug(
                "wid: {}, postfork hook called".format(uwsgi.worker_id())
            )

        UwsgiPostFork._uwsgi_broker.restful_comp().set_wid(uwsgi.worker_id())
        UwsgiPostFork._uwsgi_broker.restful_comp().set_wuuid(str(uuid.uuid4()))
        msg_prefix = "wid: {}, ".format(uwsgi.worker_id())

        uwsgi.atexit = UwsgiPostFork._uwsgi_broker.restful_comp()._on_exit

        if callable(UwsgiPostFork._postfork_op):
            UwsgiPostFork._postfork_op()
        else:
            if UwsgiPostFork._verbose:
                UwsgiPostFork._uwsgi_broker.w_logger.debug(
                    msg_prefix
                    + "'{}' is not defined by {}".format(
                        ComponentConstants.POST_FORK_CALLBACK_FUNC_NAME,
                        UwsgiPostFork._uwsgi_broker.restful_comp(),
                    )
                )

        if callable(UwsgiPostFork._configure_op):
            if UwsgiPostFork._verbose:
                UwsgiPostFork._uwsgi_broker.w_logger.debug(
                    msg_prefix + "calling configure callback ..."
                )
            UwsgiPostFork._configure_op()
        else:
            if UwsgiPostFork._verbose:
                UwsgiPostFork._uwsgi_broker.w_logger.debug(
                    msg_prefix
                    + "'{}' is not defined by {}".format(
                        ComponentConstants.CONFIGURE_CALLBACK_FUNC_NAME,
                        UwsgiPostFork._uwsgi_broker.restful_comp(),
                    )
                )

        if UwsgiPostFork._verbose:
            UwsgiPostFork._uwsgi_broker.w_logger.debug(
                msg_prefix + "calling model load callback ..."
            )
        UwsgiPostFork._uwsgi_broker._reload_last_approved_model()
