import logging
from typing import *
DEFAULT_LEVEL = logging.INFO


class GlobalLoggers:
    __logger_dict: Dict[str, logging.Logger]
    __error_list: List

    __singleton: "GlobalLoggers" = None

    @property
    def error_list(self) -> List:
        return self.__error_list

    @property
    def logger_dict(self) -> Dict[str, logging.Logger]:
        return self.__logger_dict

    def __init__(self):
        raise RuntimeError("Singleton. Use the factory function")

    def _init_helper(self):
        self.__logger_dict = dict()
        self.__error_list = []

    @classmethod
    def instance(cls) -> "GlobalLoggers":
        if cls.__singleton is None:
            cls.__singleton = cls.__new__(cls)
            cls.__singleton._init_helper()
        return cls.__singleton

    @staticmethod
    def _new_logger(name, level=DEFAULT_LEVEL) -> logging.Logger:
        logger = logging.getLogger(name)
        c_handler = logging.StreamHandler()
        # link handler to logger
        logger.addHandler(c_handler)
        logger.setLevel(level)
        return logger

    def get_logger(self, name: str, level=DEFAULT_LEVEL) -> logging.Logger:
        if name not in self.logger_dict:
            logger = GlobalLoggers._new_logger(name, level)
            self.logger_dict[name] = logger
        self.logger_dict[name].setLevel(level)
        return self.logger_dict[name]
