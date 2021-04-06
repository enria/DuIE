# coding=utf-8
import json
from collections import namedtuple


class Config(object):
    def __init__(self):
        pass

    def save_to_json_file(self, to_file):
        with open(to_file, "w") as fout:
            json.dump(self.__dict__, fout, indent=2, default=lambda x: x.__dict__, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, in_file):
        with open(in_file) as fin:
            config_dict = json.load(fin)
            return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        to_remove = []
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = cls.from_dict(value)
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
            else:
                setattr(config, key, value)

        if to_remove:
            print("Config override keys: %s" % ",".join(to_remove))

        return config

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2, default=lambda x: x.__dict__, ensure_ascii=False)


if __name__ == '__main__':
    # config = Config.from_json_file("../data/test_config.json")
    config = Config.from_dict({
        "a": 1,
        "b": {
            "c": [1, 2, 3],
            "d": {
                "x": 1
            }
        }
    })
    config.save_to_json_file("1.json")
    # print(config.__dict__)
    # print(config.a)
    # print(config.b.d.x)
    print(config)
