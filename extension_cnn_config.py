"""Configuration for the CNN-DDQN pixel-observation extension."""

from copy import deepcopy

from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


CNN_DDQN_ENV_ID = SHARED_CORE_ENV_ID


def make_cnn_ddqn_config() -> dict:
    cfg = deepcopy(SHARED_CORE_CONFIG)

    cfg["observation"] = {
        "type": "GrayscaleObservation",
        "observation_shape": (84, 84),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    }

    return cfg


CNN_DDQN_CONFIG = make_cnn_ddqn_config()
