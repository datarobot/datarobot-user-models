def str2bool(p):
    if isinstance(p, bool):
        return p

    return p.lower() in ["true", "yes", "1"]
