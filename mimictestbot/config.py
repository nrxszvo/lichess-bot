import yaml


class ConfigObj:
    def __init__(self, **entries):
        for key in entries:
            self.__dict__[key] = entries[key]


def get_config(ymlfn):
    with open(ymlfn) as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    return ConfigObj(**config)
