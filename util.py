
def get_argument(name, default, **kwargs):
    value = kwargs.get(name, default)

    assert value is not None, "{0} must be specified".format(name)

    return value


