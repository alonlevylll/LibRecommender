from enum import Enum


class FeatType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    INTERACTION_SPARSE = "interaction_sparse"
    INTERACTION_DENSE = "interaction_dense"


class Backend(Enum):
    TF = "tensorflow"
    TORCH = "torch"
