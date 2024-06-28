from mlpiper.common.singleton import Singleton


@Singleton
class VerbosePrinter:
    def __init__(self):
        self._verbose = False

    def set_verbose(self, verbose):
        self._verbose = verbose

    def verbose_print(self, msg):
        if self._verbose:
            print(msg)
