# Python imports
from dataclasses import dataclass
from typing import List, Optional, Union

# Librep imports
from librep.base.transform import Transform
from librep.base.estimator import Estimator
from librep.estimators import SVC, KNeighborsClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from librep.transforms.fft import FFT
from librep.transforms.reshaper import SimpleReshaper
from librep.transforms.spectrogram import Spectrogram
from librep.transforms.stats import StatsTransform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Third-party imports
from umap import UMAP
from librep.transforms.dimal import DIMALDimensionalityReduction
from librep.transforms.topo_ae import ConvTAETransform
from librep.transforms.lstm import LSTMTrainer
from librep.transforms.convaelstm import ConvAELSTMTransform
from librep.transforms.topo_ae_remade import TopologicalDimensionalityReductionRemade

################################################################################
# Configuration classes
################################################################################

# YAML valid confuguration keys.
# The main experiment configuration class is `ExecutionConfig`

config_version: str = "1.0"


@dataclass
class WindowedConfig:
    fit_on: Optional[str]
    transform_on: Optional[str]


@dataclass
class ReducerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict]
    use_y: Optional[bool] = False


@dataclass
class TransformConfig:
    name: str
    transform: str
    kwargs: Optional[dict] = None
    windowed: Optional[WindowedConfig] = None


@dataclass
class EstimatorConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict] = None
    num_runs: Optional[int] = 1  # Number of runs (fit/predict) for the estimator


@dataclass
class ScalerConfig:
    name: str
    algorithm: str
    kwargs: Optional[dict] = None


@dataclass
class ExtraConfig:
    in_use_features: list
    reduce_on: str  # valid values: all, sensor, axis
    scale_on: str  # valid values: self, train
    save_reducer: bool = False
    report_reducer_weight: bool = False
    report_pydrmetrics: bool = False
    # experiment_type: str = "default"

@dataclass
class MetaConfig:
    experiment_type: str = "default" # "tests_isolated"
    # baseline_comparison: bool = False # If True, compare the results with the baseline
    # baseline_improvement_rule: str = "mean" # valid values: mean, min, max

@dataclass
class ExecutionConfig:
    # Meta information
    metadata: Optional[MetaConfig]
    # control variables
    version: str
    # Datasets to use
    reducer_dataset: Optional[List[str]]
    train_dataset: List[str]
    test_dataset: List[str]
    # Transforms
    transforms: Optional[List[TransformConfig]]
    # Reducer
    reducer: Optional[ReducerConfig]
    # Scaler
    scaler: Optional[ScalerConfig]
    # Estimator
    estimators: List[EstimatorConfig]
    # Extra
    extra: ExtraConfig
    # Optional ones
    reducer_validation_dataset: Optional[List[str]] = None
    validation_dataset: Optional[List[str]] = None


################################################################################
# Transforms
################################################################################


class Identity(Transform):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def transform(self, X):
        return X


class WrapperEstimatorTransform(Estimator, Transform):
    def __init__(self, obj: Union[Estimator, Transform], *args, **kwargs) -> None:
        self._obj = obj

    def predict(self, X):
        return self._obj.predict(X)

    def transform(self, X):
        return self._obj.transform(X)


class DebugTransformEstimator(Transform, Estimator):
    def __init__(self, *args, **kwargs) -> None:
        print(f"DebugTransformEstimator: args={args}, kwargs={kwargs}")

    def fit(self, X, y=None, X_val=None, y_val=None, **fit_params):
        print(
            f"DebugTransformEstimator (fit). X: {len(X)}, y: {len(y) if y is not None else None}, X_val: {len(X_val) if X_val is not None else None}, y_val: {len(y_val) if y_val is not None else None}, fit_params: {fit_params}"
        )
        return self

    def transform(self, X):
        print(f"DebugTransformEstimator (transform): {X.shape}")
        return X
    
    def predict(self, X):
        print(f"DebugTransformEstimator (predict): {X.shape}")
        return np.zeros(len(X))


################################################################################
# Constants (Valid keys)
################################################################################

# Dictionary with the valid estimators keys to use in experiment configuration
# (under estimator.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Estimators must be a subclass of `librep.estimators.base.BaseEstimator` or implement
# the same interface (scikit-learn compatible, fit/predict methods)
estimator_cls = {
    "SVM": SVC,
    "KNN": KNeighborsClassifier,
    "RandomForest": RandomForestClassifier,
    "DebugEstimator": DebugTransformEstimator,
    "WrapperEstimator": WrapperEstimatorTransform,
    "MLP": MLPClassifier,
}

# Dictionary with the valid reducer keys to use in experiment configuration
# (under reducer.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Reducers must be a subclass of `librep.reducers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
reducers_cls = {
    "identity": Identity,
    "umap": UMAP,
    "pca": PCA,
    "WrapperTransform": WrapperEstimatorTransform,
    "DebugReducer": DebugTransformEstimator,
    "dimal": DIMALDimensionalityReduction,
    "convtae1d": ConvTAETransform,
    "lstm": LSTMTrainer,
    "convaelstm": ConvAELSTMTransform,
    'convtae1dremade': TopologicalDimensionalityReductionRemade
}
# Dictionary with the valid transforms keys to use in experiment configuration
# (under transform.transform key).
# The key is the algorithm name and the value is the class to use.
# Transforms must be a subclass of `librep.transforms.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
transforms_cls = {
    "identity": Identity,
    "fft": FFT,
    "WrapperTransform": WrapperEstimatorTransform,
    "DebugTransform": DebugTransformEstimator,
    "spectrogram": Spectrogram,
    "stats": StatsTransform,
    "reshaper": SimpleReshaper
}

# Dictionary with the valid scalers keys to use in experiment configuration
# (under scaler.algorithm key).
# The key is the algorithm name and the value is the class to use.
# Scalers must be a subclass of `librep.scalers.base.Transform` or implement
# the same interface (scikit-learn compatible, fit/transform methods)
scaler_cls = {
    "identity": Identity,
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

# Dictionary with standard labels for each activity code
standard_labels_activity = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down",
}
