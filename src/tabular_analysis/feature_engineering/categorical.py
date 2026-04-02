from __future__ import annotations
from ..common.config_utils import normalize_str as _normalize_str
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
def normalize_encoding(value: Any) -> str:
    key = (_normalize_str(value) or '').lower()
    if key in ('ohe', 'one_hot', 'one_hot_encoder'):
        return 'onehot'
    if key in ('hash', 'hashing', 'hasher'):
        return 'hashing'
    if key in ('freq', 'frequency', 'count'):
        return 'frequency'
    if key in ('target_mean_oof', 'target_oof', 'oof_mean', 'target_mean', 'mean_target'):
        return 'target_mean_oof'
    if key in ('ordinal', 'ordinal_encoder'):
        return 'ordinal'
    return key
def _to_frame(data: Any, *, columns: Sequence[str] | None=None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data, columns=list(columns) if columns is not None else None)
def _as_array(data: Any) -> np.ndarray:
    if hasattr(data, 'toarray'):
        return data.toarray()
    return np.asarray(data)
def _concat(parts: Sequence[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty((0, 0))
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=1)
def _fit_encoder(encoder: Any, X, y: Any | None=None) -> Any:
    if y is None:
        return encoder.fit(X)
    try:
        return encoder.fit(X, y)
    except TypeError:
        return encoder.fit(X)
def _get_feature_names(encoder: Any, input_features: Sequence[str]) -> list[str]:
    if hasattr(encoder, 'get_feature_names_out'):
        try:
            names = encoder.get_feature_names_out(input_features)
        except TypeError:
            names = encoder.get_feature_names_out()
        return [str(name) for name in list(names)]
    return [str(name) for name in input_features]
def _stringify(value: Any) -> str:
    if value is None:
        return '__MISSING__'
    try:
        if pd.isna(value):
            return '__MISSING__'
    except (TypeError, ValueError):
        pass
    return str(value)
class FrequencyEncoder:
    def __init__(self) -> None:
        self.columns_: list[str] = []
        self.frequencies_: dict[str, dict[Any, float]] = {}
    def fit(self, X, y: Any | None=None) -> 'FrequencyEncoder':
        df = _to_frame(X)
        self.columns_ = [str(col) for col in df.columns]
        total = float(len(df)) if len(df) > 0 else 1.0
        frequencies: dict[str, dict[Any, float]] = {}
        for col in self.columns_:
            counts = df[col].value_counts(dropna=False)
            frequencies[col] = {key: float(count / total) for (key, count) in counts.items()}
        self.frequencies_ = frequencies
        return self
    def transform(self, X) -> np.ndarray:
        df = _to_frame(X, columns=self.columns_)
        result = np.zeros((len(df), len(self.columns_)), dtype=float)
        for (idx, col) in enumerate(self.columns_):
            mapping = self.frequencies_.get(col, {})
            result[:, idx] = df[col].map(mapping).fillna(0.0).astype(float)
        return result
    def get_feature_names_out(self, input_features: Sequence[str] | None=None) -> np.ndarray:
        features = list(input_features) if input_features is not None else list(self.columns_)
        return np.asarray([str(name) for name in features], dtype=object)
class HashingEncoder:
    def __init__(self, *, n_features: int, alternate_sign: bool=False) -> None:
        self.n_features = int(n_features)
        self.alternate_sign = bool(alternate_sign)
        self.columns_: list[str] = []
        self._hashers: list[FeatureHasher] = []
    def fit(self, X, y: Any | None=None) -> 'HashingEncoder':
        if self.n_features <= 0:
            raise ValueError('hashing.n_features must be positive.')
        df = _to_frame(X)
        self.columns_ = [str(col) for col in df.columns]
        self._hashers = [FeatureHasher(n_features=self.n_features, input_type='string', alternate_sign=self.alternate_sign) for _ in self.columns_]
        return self
    def transform(self, X) -> np.ndarray:
        df = _to_frame(X, columns=self.columns_)
        parts: list[np.ndarray] = []
        for (idx, col) in enumerate(self.columns_):
            tokens = [f'{col}={_stringify(value)}' for value in df[col].tolist()]
            samples = [[token] for token in tokens]
            hashed = self._hashers[idx].transform(samples)
            parts.append(_as_array(hashed).astype(float))
        return _concat(parts)
    def get_feature_names_out(self, input_features: Sequence[str] | None=None) -> np.ndarray:
        columns = list(input_features) if input_features is not None else list(self.columns_)
        names: list[str] = []
        for col in columns:
            for idx in range(self.n_features):
                names.append(f'{col}__hash_{idx}')
        return np.asarray(names, dtype=object)
class TargetMeanEncoder:
    def __init__(self, *, smoothing: float | None=10.0) -> None:
        self.smoothing = float(smoothing) if smoothing is not None else 0.0
        self.columns_: list[str] = []
        self.mapping_: dict[str, dict[Any, float]] = {}
        self.global_mean_: float = 0.0
    def fit(self, X, y: Any | None=None) -> 'TargetMeanEncoder':
        if y is None:
            raise ValueError('TargetMeanEncoder requires y for fit.')
        df = _to_frame(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(df) != len(y_arr):
            raise ValueError('TargetMeanEncoder.fit requires X and y with the same length.')
        self.columns_ = [str(col) for col in df.columns]
        (self.mapping_, self.global_mean_) = _compute_target_mean_mapping(df, y_arr, smoothing=self.smoothing)
        return self
    def transform(self, X) -> np.ndarray:
        df = _to_frame(X, columns=self.columns_)
        return _apply_target_mean(df, self.mapping_, self.global_mean_)
    def get_feature_names_out(self, input_features: Sequence[str] | None=None) -> np.ndarray:
        features = list(input_features) if input_features is not None else list(self.columns_)
        return np.asarray([str(name) for name in features], dtype=object)
class CompositeCategoricalEncoder:
    def __init__(self, encoders: Sequence[tuple[str, Any, Sequence[str]]]) -> None:
        self.encoders = list(encoders)
        self.columns_: list[str] = []
    def fit(self, X, y: Any | None=None) -> 'CompositeCategoricalEncoder':
        df = _to_frame(X)
        self.columns_ = [str(col) for col in df.columns]
        for (_, encoder, columns) in self.encoders:
            cols = list(columns)
            if not cols:
                continue
            _fit_encoder(encoder, df[cols], y)
        return self
    def transform(self, X) -> np.ndarray:
        df = _to_frame(X, columns=self.columns_)
        parts: list[np.ndarray] = []
        for (_, encoder, columns) in self.encoders:
            cols = list(columns)
            if not cols:
                continue
            part = encoder.transform(df[cols])
            parts.append(_as_array(part).astype(float))
        return _concat(parts)
    def get_feature_names_out(self, input_features: Sequence[str] | None=None) -> np.ndarray:
        names: list[str] = []
        for (_, encoder, columns) in self.encoders:
            cols = list(columns)
            if not cols:
                continue
            names.extend(_get_feature_names(encoder, cols))
        return np.asarray(names, dtype=object)
class TabularPreprocessor:
    def __init__(self, *, numeric_features: Sequence[str], categorical_features: Sequence[str], numeric_pipeline: Pipeline | None, categorical_imputer: SimpleImputer | None, categorical_encoder: Any | None) -> None:
        self.numeric_features = [str(col) for col in numeric_features]
        self.categorical_features = [str(col) for col in categorical_features]
        self.numeric_pipeline = numeric_pipeline
        self.categorical_imputer = categorical_imputer
        self.categorical_encoder = categorical_encoder
        self._feature_names: list[str] | None = None
    def fit(self, X, y: Any | None=None) -> 'TabularPreprocessor':
        df = _to_frame(X)
        if self.numeric_features:
            if self.numeric_pipeline is None:
                raise ValueError('numeric_pipeline is required for numeric features.')
            self.numeric_pipeline.fit(df[self.numeric_features])
        if self.categorical_features:
            if self.categorical_imputer is None or self.categorical_encoder is None:
                raise ValueError('categorical_imputer/encoder are required for categorical features.')
            cat = self.categorical_imputer.fit_transform(df[self.categorical_features])
            cat_df = _to_frame(cat, columns=self.categorical_features)
            _fit_encoder(self.categorical_encoder, cat_df, y)
        self._feature_names = self.get_feature_names_out()
        return self
    def transform(self, X) -> np.ndarray:
        df = _to_frame(X)
        parts: list[np.ndarray] = []
        if self.numeric_features:
            if self.numeric_pipeline is None:
                raise ValueError('numeric_pipeline is required for numeric features.')
            parts.append(_as_array(self.numeric_pipeline.transform(df[self.numeric_features])).astype(float))
        if self.categorical_features:
            if self.categorical_imputer is None or self.categorical_encoder is None:
                raise ValueError('categorical_imputer/encoder are required for categorical features.')
            cat = self.categorical_imputer.transform(df[self.categorical_features])
            cat_df = _to_frame(cat, columns=self.categorical_features)
            encoded = self.categorical_encoder.transform(cat_df)
            parts.append(_as_array(encoded).astype(float))
        return _concat(parts)
    def transform_with_oof(self, X, y, *, train_idx: Sequence[int], folds: int, seed: int, task_type: str) -> np.ndarray:
        if not isinstance(self.categorical_encoder, TargetMeanEncoder) or not self.categorical_features:
            return self.transform(X)
        df = _to_frame(X)
        parts: list[np.ndarray] = []
        if self.numeric_features:
            if self.numeric_pipeline is None:
                raise ValueError('numeric_pipeline is required for numeric features.')
            parts.append(_as_array(self.numeric_pipeline.transform(df[self.numeric_features])).astype(float))
        if self.categorical_imputer is None:
            raise ValueError('categorical_imputer is required for target mean encoding.')
        cat = self.categorical_imputer.transform(df[self.categorical_features])
        cat_df = _to_frame(cat, columns=self.categorical_features)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        train_idx_array = np.asarray(list(train_idx), dtype=int)
        oof_encoded = target_mean_oof_encode(cat_df.iloc[train_idx_array], y_arr[train_idx_array], folds=folds, smoothing=self.categorical_encoder.smoothing, seed=seed, classification=task_type == 'classification')
        encoded = self.categorical_encoder.transform(cat_df)
        encoded = _as_array(encoded).astype(float)
        encoded[train_idx_array] = oof_encoded
        parts.append(encoded)
        return _concat(parts)
    def get_feature_names_out(self) -> list[str]:
        names: list[str] = []
        if self.numeric_features:
            names.extend([f'num__{col}' for col in self.numeric_features])
        if self.categorical_features and self.categorical_encoder is not None:
            enc_names = _get_feature_names(self.categorical_encoder, self.categorical_features)
            names.extend([f'cat__{name}' for name in enc_names])
        return names
def _compute_target_mean_mapping(df: pd.DataFrame, y: np.ndarray, *, smoothing: float) -> tuple[dict[str, dict[Any, float]], float]:
    smoothing_value = float(smoothing) if smoothing is not None else 0.0
    smoothing_value = max(smoothing_value, 0.0)
    global_mean = float(np.mean(y)) if len(y) else 0.0
    mapping: dict[str, dict[Any, float]] = {}
    for col in df.columns:
        stats = pd.DataFrame({'key': df[col], 'y': y}).groupby('key')['y'].agg(['mean', 'count'])
        smoothed = (stats['count'] * stats['mean'] + smoothing_value * global_mean) / (stats['count'] + smoothing_value)
        mapping[str(col)] = smoothed.to_dict()
    return (mapping, global_mean)
def _apply_target_mean(df: pd.DataFrame, mapping: Mapping[str, Mapping[Any, float]], global_mean: float) -> np.ndarray:
    result = np.zeros((len(df), len(df.columns)), dtype=float)
    for (idx, col) in enumerate(df.columns):
        col_map = mapping.get(str(col), {})
        result[:, idx] = df[col].map(col_map).fillna(global_mean).astype(float)
    return result
def target_mean_oof_encode(X, y, *, folds: int, smoothing: float | None, seed: int, classification: bool) -> np.ndarray:
    df = _to_frame(X)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    n_samples = len(df)
    if folds < 2:
        raise ValueError('target_mean_oof.folds must be >= 2.')
    if n_samples < folds:
        raise ValueError('target_mean_oof.folds must be <= number of training samples.')
    if classification:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n_samples), y_arr)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n_samples))
    encoded = np.zeros((n_samples, len(df.columns)), dtype=float)
    for (train_idx, val_idx) in split_iter:
        (mapping, global_mean) = _compute_target_mean_mapping(df.iloc[train_idx], y_arr[train_idx], smoothing=float(smoothing or 0.0))
        encoded[val_idx] = _apply_target_mean(df.iloc[val_idx], mapping, global_mean)
    return encoded
def encode_target_for_mean(train_values: Iterable[Any], all_values: Iterable[Any], *, task_type: str) -> tuple[np.ndarray, dict[str, Any]]:
    if task_type != 'classification':
        return (np.asarray(list(all_values), dtype=float), {'task_type': task_type, 'n_classes': None})
    try:
        from sklearn.preprocessing import LabelEncoder
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError('scikit-learn is required for target_mean_oof encoding.') from exc
    encoder = LabelEncoder()
    encoder.fit_transform(list(train_values))
    y_all = encoder.transform(list(all_values))
    n_classes = int(len(encoder.classes_))
    if n_classes != 2:
        raise ValueError('target_mean_oof supports regression or binary classification only.')
    return (y_all.astype(float), {'task_type': task_type, 'n_classes': n_classes, 'class_labels': [str(v) for v in encoder.classes_]})
def build_tabular_preprocessor(preprocess_variant: Mapping[str, Any], *, numeric_features: Sequence[str], categorical_features: Sequence[str], numeric_impute: str, categorical_impute: str, categorical_encoding: str, auto_onehot_max_categories: int, hashing_n_features: int, target_mean_smoothing: float | None, unique_counts: Mapping[str, int] | None) -> tuple[TabularPreprocessor, dict[str, str], str]:
    if not numeric_features and (not categorical_features):
        raise ValueError('No feature columns available for preprocessing.')
    numeric_scaler = (_normalize_str(preprocess_variant.get('numeric_scaler', 'standard')) or '').lower()
    handle_unknown = preprocess_variant.get('handle_unknown', 'ignore')
    numeric_steps = [('imputer', SimpleImputer(strategy=str(numeric_impute)))]
    if numeric_scaler in ('', 'none', 'passthrough'):
        pass
    elif numeric_scaler in ('standard', 'std', 'stdscaler'):
        numeric_steps.append(('scaler', StandardScaler()))
    elif numeric_scaler in ('minmax', 'min_max', 'minmaxscaler'):
        numeric_steps.append(('scaler', MinMaxScaler()))
    else:
        raise ValueError(f'Unsupported numeric_scaler: {numeric_scaler}')
    numeric_pipeline = Pipeline(steps=numeric_steps) if numeric_features else None
    categorical_imputer = SimpleImputer(strategy=str(categorical_impute)) if categorical_features else None
    encoding = normalize_encoding(categorical_encoding)
    if not encoding:
        encoding = normalize_encoding(preprocess_variant.get('categorical_encoder', 'onehot')) or 'onehot'
    encoding_by_column: dict[str, str] = {}
    if encoding == 'auto':
        if unique_counts is None:
            raise ValueError('auto encoding requires unique_counts.')
        onehot_cols: list[str] = []
        hash_cols: list[str] = []
        for col in categorical_features:
            unique = int(unique_counts.get(col, 0))
            if unique <= int(auto_onehot_max_categories):
                encoding_by_column[str(col)] = 'onehot'
                onehot_cols.append(str(col))
            else:
                encoding_by_column[str(col)] = 'hashing'
                hash_cols.append(str(col))
        encoders: list[tuple[str, Any, Sequence[str]]] = []
        if onehot_cols:
            encoders.append(('onehot', OneHotEncoder(handle_unknown=str(handle_unknown), sparse_output=False), onehot_cols))
        if hash_cols:
            encoders.append(('hashing', HashingEncoder(n_features=hashing_n_features), hash_cols))
        categorical_encoder: Any | None = CompositeCategoricalEncoder(encoders) if encoders else None
    else:
        encoding_by_column = {str(col): encoding for col in categorical_features}
        if encoding in ('onehot', 'ohe'):
            categorical_encoder = OneHotEncoder(handle_unknown=str(handle_unknown), sparse_output=False)
        elif encoding in ('ordinal', 'ordinal_encoder'):
            categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        elif encoding == 'frequency':
            categorical_encoder = FrequencyEncoder()
        elif encoding == 'hashing':
            categorical_encoder = HashingEncoder(n_features=hashing_n_features)
        elif encoding == 'target_mean_oof':
            categorical_encoder = TargetMeanEncoder(smoothing=target_mean_smoothing)
        else:
            raise ValueError(f'Unsupported categorical encoding: {encoding}')
    preprocessor = TabularPreprocessor(numeric_features=numeric_features, categorical_features=categorical_features, numeric_pipeline=numeric_pipeline, categorical_imputer=categorical_imputer, categorical_encoder=categorical_encoder)
    return (preprocessor, encoding_by_column, encoding)
def build_categorical_encoding_report(categorical_features: Sequence[str], *, encoding: str, encoding_by_column: Mapping[str, str], unique_counts: Mapping[str, int], hashing_n_features: int) -> dict[str, Any]:
    columns_report: dict[str, dict[str, Any]] = {}
    total_dim = 0
    for col in categorical_features:
        col_name = str(col)
        col_encoding = encoding_by_column.get(col_name, encoding)
        unique = int(unique_counts.get(col_name, 0))
        if col_encoding == 'hashing':
            encoded_dim = int(hashing_n_features)
        elif col_encoding in ('frequency', 'target_mean_oof', 'ordinal'):
            encoded_dim = 1
        else:
            encoded_dim = unique
        total_dim += encoded_dim
        columns_report[col_name] = {'encoding': col_encoding, 'unique': unique, 'encoded_dim': encoded_dim}
    return {'encoding': encoding, 'columns': columns_report, 'total_encoded_dim': total_dim}
