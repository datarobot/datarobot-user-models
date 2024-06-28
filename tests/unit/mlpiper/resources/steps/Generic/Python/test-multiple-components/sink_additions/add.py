from sink_additions.util import util


def sink_decode(str):
    util.show_py_ver()
    return "".join(chr(ord(letter) - 1) for letter in str)
