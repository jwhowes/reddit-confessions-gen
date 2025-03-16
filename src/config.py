import os
from abc import ABC
from typing import Self

import yaml


class BaseConfig(ABC):
    @classmethod
    def from_yaml(cls, config_path: str) -> Self:
        if not os.path.exists(config_path):
            return cls()

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return cls(**{
            k: float(v) if cls.__dataclass_fields__[k] == float else v
            for k, v in config.items() if k in cls.__dataclass_fields__
        })
