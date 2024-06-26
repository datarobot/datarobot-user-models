def source_encode(s):
    return "".join(chr(ord(letter) + 1) for letter in s)
