from contextlib import contextmanager


@contextmanager
def capture_R_traceback_if_errors(r_handler, logger):
    from rpy2.rinterface_lib.embedded import RRuntimeError

    try:
        yield
    except RRuntimeError as e:
        try:
            out = "\n".join(r_handler("capture.output(traceback(max.lines = 50))"))
            logger.error("R Traceback:\n{}".format(str(out)))
        except Exception as traceback_exc:
            e.context = {
                "r_traceback": "(an error occurred while getting traceback from R)",
                "t_traceback_err": traceback_exc,
            }
        raise
