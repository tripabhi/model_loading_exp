import os


def get_env(key, default_value=""):
    __env = os.getenv(key)
    return __env if __env is not None else default_value


def get_model_store_path():
    __data_dir = get_env("DATA_DIR", "model_store")
    __base_dir = os.path.dirname(__file__)
    return os.path.join(__base_dir, __data_dir)
