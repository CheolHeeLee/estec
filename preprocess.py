import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset


DATETIME_COLUMN = "datetime"
WEEKDAY_COLUMN = "weekday"
FEATURE_COLUMNS = ["power_usage", "flow_usage"]
LABEL_COLUMN = "idle_time"
CLASS_NAMES = {0: "휴지", 1: "작동"}
DAY_NAMES = {
    1: "월",
    2: "화",
    3: "수",
    4: "목",
    5: "금",
    6: "토",
    7: "일",
}


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_activity_csv(path: str, is_inference: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    req_cols = [DATETIME_COLUMN, WEEKDAY_COLUMN] + FEATURE_COLUMNS
    if not is_inference:
        req_cols.append(LABEL_COLUMN)
        
    missing = [col for col in req_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{path}의 컬럼 형식이 맞지 않습니다. 누락 컬럼: {missing}"
        )

    df = df[req_cols].copy()
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors="raise")
    df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    df[WEEKDAY_COLUMN] = pd.to_numeric(df[WEEKDAY_COLUMN], errors="raise")
    df["power_usage"] = pd.to_numeric(df["power_usage"], errors="raise")
    df["flow_usage"] = pd.to_numeric(df["flow_usage"], errors="raise")
    
    if not is_inference:
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="raise")

    if df.isna().any().any():
        raise ValueError(f"{path}에 결측값이 있습니다.")

    for column in [WEEKDAY_COLUMN] + FEATURE_COLUMNS:
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}의 {column}에 유한하지 않은 값이 있습니다.")

    weekday_values = df[WEEKDAY_COLUMN].to_numpy(dtype=float)
    if not np.equal(weekday_values, np.floor(weekday_values)).all():
        raise ValueError(f"{path}의 weekday는 정수여야 합니다.")

    df[WEEKDAY_COLUMN] = df[WEEKDAY_COLUMN].astype(int)
    if not is_inference:
        label_values = df[LABEL_COLUMN].to_numpy(dtype=float)
        if not np.isfinite(label_values).all():
            raise ValueError(f"{path}의 idle_time에 유한하지 않은 값이 있습니다.")
        if not np.equal(label_values, np.floor(label_values)).all():
            raise ValueError(f"{path}의 idle_time은 정수여야 합니다.")
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
        invalid_labels = sorted(set(df[LABEL_COLUMN]) - {0, 1})
        if invalid_labels:
            raise ValueError(f"{path}의 idle_time은 0 또는 1이어야 합니다.")

    invalid_days = sorted(set(df[WEEKDAY_COLUMN]) - set(DAY_NAMES))
    if invalid_days:
        raise ValueError(f"{path}의 weekday는 1~7이어야 합니다.")

    return df


def _continuous_datetime_groups(data: pd.DataFrame, expected_interval: pd.Timedelta):
    sorted_data = data.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    gap = sorted_data[DATETIME_COLUMN].diff()
    segment_ids = gap.ne(expected_interval).cumsum()
    return sorted_data.groupby(segment_ids, sort=False)


def create_sequences_per_group(
    data: pd.DataFrame,
    seq_length: int,
    forecast_horizon_steps: int = 1,
):
    return create_sequences(
        data,
        seq_length,
        forecast_horizon_steps,
        target_column=LABEL_COLUMN,
    )


def create_sequences(
    data: pd.DataFrame,
    seq_length: int,
    forecast_horizon_steps: int = 1,
    target_column: str = None,
):
    X, targets, datetimes = [], [], []

    continuous_groups = _continuous_datetime_groups(data, pd.Timedelta(minutes=1))
    for _, group in continuous_groups:
        if len(group) < seq_length + forecast_horizon_steps:
            continue

        features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = (
            group[target_column].to_numpy(dtype=np.int64)
            if target_column is not None
            else None
        )
        times = group[DATETIME_COLUMN].to_numpy()

        for i in range(len(group) - seq_length - forecast_horizon_steps + 1):
            target_pos = i + seq_length + forecast_horizon_steps - 1
            X.append(features[i : i + seq_length])
            if labels is not None:
                targets.append(labels[target_pos])
            datetimes.append(times[target_pos])

    if not X:
        empty_X = np.empty((0, seq_length, len(FEATURE_COLUMNS)), dtype=np.float32)
        empty_datetimes = np.empty((0,), dtype="datetime64[ns]")
        if target_column is None:
            return empty_X, empty_datetimes
        return empty_X, np.empty((0,), dtype=np.int64), empty_datetimes

    X_array = np.asarray(X, dtype=np.float32)
    datetime_array = np.asarray(datetimes, dtype="datetime64[ns]")
    if target_column is None:
        return X_array, datetime_array
    return X_array, np.asarray(targets, dtype=np.int64), datetime_array


def build_scaler_fingerprint(scaler: RobustScaler, seq_length: int) -> str:
    """스케일러와 입력 계약이 같은 학습 세대인지 확인할 식별자를 만듭니다."""
    digest = hashlib.sha256()
    digest.update("|".join(FEATURE_COLUMNS).encode("utf-8"))
    digest.update(str(seq_length).encode("ascii"))
    for attribute in ("center_", "scale_"):
        values = np.asarray(getattr(scaler, attribute), dtype=np.float64)
        digest.update(values.tobytes())
    return digest.hexdigest()


def create_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    X_test=None,
    y_test=None,
):
    loaders = {
        "train_loader": DataLoader(
            TimeSeriesDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        ),
        "val_loader": DataLoader(
            TimeSeriesDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    if X_test is not None and y_test is not None:
        loaders["test_loader"] = DataLoader(
            TimeSeriesDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )
    return loaders


def preprocess(
    train_path,
    test_path=None,
    seq_length=60,
    batch_size=64,
    validation_ratio=0.2,
):
    print("=" * 60)
    print("전처리 시작")
    print("=" * 60)

    train_df = load_activity_csv(train_path)
    test_df = load_activity_csv(test_path) if test_path is not None else None
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio는 0과 1 사이여야 합니다.")

    split_index = int(len(train_df) * (1 - validation_ratio))
    if split_index < seq_length or len(train_df) - split_index < seq_length:
        raise ValueError("학습/검증 데이터를 만들기에 train.csv가 너무 작습니다.")
    validation_start = train_df.iloc[split_index][DATETIME_COLUMN]
    fit_df = train_df.iloc[:split_index].copy()
    print(f"학습 데이터: {len(train_df):,}행")
    print(f"검증 데이터: {len(train_df) - split_index:,}행 (train.csv의 마지막 {validation_ratio:.0%})")
    if test_df is not None:
        print(f"테스트 데이터: {len(test_df):,}행 (최종 평가 전용)")
    else:
        print("테스트 데이터: 학습 과정에서 읽지 않음")

    scalers = {}
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    from sklearn.preprocessing import RobustScaler
    for day_value in sorted(train_df[WEEKDAY_COLUMN].unique()):
        scaler = RobustScaler()
        
        fit_mask = fit_df[WEEKDAY_COLUMN] == day_value
        if fit_mask.any():
            scaler.fit(fit_df.loc[fit_mask, FEATURE_COLUMNS])
            scaler.yms_fingerprint_ = build_scaler_fingerprint(scaler, seq_length)
            train_mask = train_df[WEEKDAY_COLUMN] == day_value
            train_scaled.loc[train_mask, FEATURE_COLUMNS] = scaler.transform(train_df.loc[train_mask, FEATURE_COLUMNS])
            scalers[day_value] = scaler

        test_mask = (
            test_df[WEEKDAY_COLUMN] == day_value
            if test_df is not None
            else None
        )
        if test_mask is not None and test_mask.any() and day_value in scalers:
            test_scaled.loc[test_mask, FEATURE_COLUMNS] = scalers[day_value].transform(test_df.loc[test_mask, FEATURE_COLUMNS])

    X_train_full, y_train_full, dt_train_full = create_sequences_per_group(train_scaled, seq_length)
    if test_scaled is not None:
        X_test_full, y_test_full, dt_test_full = create_sequences_per_group(test_scaled, seq_length)

    weekday_data = {}
    
    for day_value in sorted(train_df[WEEKDAY_COLUMN].unique()):
        day_name = DAY_NAMES.get(int(day_value), str(day_value))
        
        train_weekdays = pd.to_datetime(dt_train_full).dayofweek + 1
        test_weekdays = (
            pd.to_datetime(dt_test_full).dayofweek + 1
            if test_scaled is not None
            else None
        )
        
        train_idx = (train_weekdays == day_value) & (dt_train_full < validation_start)
        val_idx = (train_weekdays == day_value) & (dt_train_full >= validation_start)
        test_idx = (test_weekdays == day_value) if test_weekdays is not None else None
        
        day_X_train, day_y_train, day_dt_train = X_train_full[train_idx], y_train_full[train_idx], dt_train_full[train_idx]
        day_X_val, day_y_val, day_dt_val = X_train_full[val_idx], y_train_full[val_idx], dt_train_full[val_idx]
        if test_idx is not None:
            day_X_test = X_test_full[test_idx]
            day_y_test = y_test_full[test_idx]
            day_dt_test = dt_test_full[test_idx]

        if len(day_X_train) == 0 or len(day_X_val) == 0:
            print(f"[day={day_value}] 학습/검증 시퀀스 부족")
            continue
        if test_idx is not None and len(day_X_test) == 0:
            print(f"[day={day_value}] 테스트 시퀀스 부족")
            continue
            
        loaders = create_dataloaders(
            day_X_train, day_y_train,
            day_X_val, day_y_val,
            batch_size,
            day_X_test if test_idx is not None else None,
            day_y_test if test_idx is not None else None,
        )

        data = {
            **loaders,
            "X_train": day_X_train,
            "y_train": day_y_train,
            "train_datetime": day_dt_train,
            "X_val": day_X_val,
            "y_val": day_y_val,
            "val_datetime": day_dt_val,
            "train_count": len(day_X_train),
            "input_size": len(FEATURE_COLUMNS),
            "seq_length": seq_length,
            "forecast_horizon_steps": 1,
            "num_classes": 2,
            "task": "classification",
            "day_name": day_name,
            "day_value": day_value,
            "scaler": scalers[day_value],
            "scaler_fingerprint": scalers[day_value].yms_fingerprint_,
        }
        if test_idx is not None:
            data.update(
                {
                    "X_test": day_X_test,
                    "y_test": day_y_test,
                    "test_datetime": day_dt_test,
                    "test_count": len(day_X_test),
                }
            )
        weekday_data[day_value] = data

        message = f"[{day_name}] 학습: {day_X_train.shape}, 검증: {day_X_val.shape}"
        if test_idx is not None:
            message += f", 테스트: {day_X_test.shape}"
        print(message)

    if not weekday_data:
        raise ValueError("학습 가능한 day 그룹이 없습니다.")

    return weekday_data
