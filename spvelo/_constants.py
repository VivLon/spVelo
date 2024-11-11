from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    U_KEY: str = "U"
    INDEX_KEY: str = "INDEX"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "cell_type"
    SPATIAL_KEY: str = "spatial"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
