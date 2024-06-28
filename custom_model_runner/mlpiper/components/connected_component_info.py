class ConnectedComponentInfo(object):
    """
    Information about a connected component, which is needed in order to run the component,
    and provide information to child components.
    """

    def __init__(self):
        self.params = {}
        self.parents_objs = []
        self.output_objs = []

    def __str__(self):
        s = "Info:\n"
        s += "params:       {}\n".format(self.params)
        s += "parents_obj:  {}\n".format(self.parents_objs)
        s += "output_obj:   {}\n".format(self.output_objs)
        return s
