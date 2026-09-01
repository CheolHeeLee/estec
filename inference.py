import os
import warnings
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Tuple

from model import load_model_from_checkpoint
from preprocess import (
    create_sequences,
    load_activity_csv,
    FEATURE_COLUMNS, 
    WEEKDAY_COLUMN, 
    DAY_NAMES,
)

class CompressorPredictor:
    
    def __init__(self, checkpoint_dir: str, use_union_model: bool = False, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_dir = checkpoint_dir
        self.use_union_model = use_union_model
        
        self.models = {}
        self.scalers = {}
        self.seq_length = 60
        
        # 요일별 스케일러 로드 
        missing_scalers = []
        for day, day_name in DAY_NAMES.items():
            scaler_path = os.path.join(checkpoint_dir, f"scaler_{day}_{day_name}.joblib")
            if os.path.exists(scaler_path):
                self.scalers[day] = joblib.load(scaler_path)
            else:
                missing_scalers.append(scaler_path)

        if missing_scalers:
            raise FileNotFoundError(
                "필수 스케일러 파일이 없습니다: " + ", ".join(missing_scalers)
            )
                
        # 모델 로드 
        if use_union_model:
            union_ckpt = os.path.join(checkpoint_dir, "best_lstm_model_union.pth")
            if not os.path.exists(union_ckpt):
                raise FileNotFoundError(f"필수 모델 파일이 없습니다: {union_ckpt}")
            self.models["union"] = self._load_model(union_ckpt)
        else:
            missing_models = []
            for day, day_name in DAY_NAMES.items():
                ckpt_path = os.path.join(checkpoint_dir, f"best_lstm_model_{day}_{day_name}.pth")
                if os.path.exists(ckpt_path):
                    self.models[day] = self._load_model(ckpt_path)
                else:
                    missing_models.append(ckpt_path)
            if missing_models:
                raise FileNotFoundError(
                    "필수 모델 파일이 없습니다: " + ", ".join(missing_models)
                )

        sequence_lengths = {
            model.checkpoint_metadata["seq_length"]
            for model in self.models.values()
        }
        if len(sequence_lengths) != 1:
            raise ValueError(f"모델별 seq_length가 서로 다릅니다: {sorted(sequence_lengths)}")
        self.seq_length = sequence_lengths.pop()
        self._validate_artifact_compatibility()

    def _load_model(self, path: str):
        model, config = load_model_from_checkpoint(path, self.device)
        if config.get("input_size") != len(FEATURE_COLUMNS):
            raise ValueError(
                f"모델 input_size가 현재 피처 수와 다릅니다: {path}"
            )
        model.checkpoint_metadata["seq_length"] = config.get("seq_length", 60)
        return model

    def _validate_artifact_compatibility(self):
        if self.use_union_model:
            expected = self.models["union"].checkpoint_metadata.get("scaler_fingerprints")
            pairs = [
                (
                    day,
                    (expected.get(day) or expected.get(str(day)))
                    if expected
                    else None,
                )
                for day in DAY_NAMES
            ]
        else:
            pairs = [
                (day, self.models[day].checkpoint_metadata.get("scaler_fingerprint"))
                for day in DAY_NAMES
            ]

        legacy_days = []
        for day, expected_fingerprint in pairs:
            actual_fingerprint = getattr(self.scalers[day], "yms_fingerprint_", None)
            if expected_fingerprint is None and actual_fingerprint is None:
                legacy_days.append(day)
                continue
            if expected_fingerprint is None or actual_fingerprint is None:
                raise ValueError(
                    f"{day}요일 모델과 스케일러 중 한쪽에만 전처리 버전 정보가 있습니다."
                )
            if expected_fingerprint != actual_fingerprint:
                raise ValueError(
                    f"{day}요일 모델과 스케일러가 서로 다른 전처리 버전입니다."
                )

        if legacy_days:
            warnings.warn(
                "기존 체크포인트에는 스케일러 호환성 정보가 없어 검증할 수 없습니다. "
                "새 코드로 재학습하면 엄격한 검증이 활성화됩니다. "
                f"대상 요일: {legacy_days}",
                RuntimeWarning,
            )

    def _create_inference_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return create_sequences(df, self.seq_length)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
        df_scaled = df.copy()
        for day_value, day_df in df.groupby(WEEKDAY_COLUMN, sort=False):
            if day_value in self.scalers:
                df_scaled.loc[day_df.index, FEATURE_COLUMNS] = self.scalers[day_value].transform(day_df[FEATURE_COLUMNS])
            else:
                raise RuntimeError(f"{day_value}요일 스케일러가 로드되지 않았습니다.")
                
        X_seq, target_times = self._create_inference_sequences(df_scaled)
        if len(X_seq) == 0:
            raise ValueError(
                f"예측 시퀀스를 만들 수 없습니다. 최소 {self.seq_length + 1}개의 연속 데이터가 필요합니다."
            )

        target_dates = pd.to_datetime(target_times)
        target_weekdays = target_dates.dayofweek + 1 
        
        results = []
        
        if self.use_union_model:
            model = self.models.get("union")
            if model:
                results.append(self._run_inference(model, X_seq, target_times, batch_size))
        else:
            for day_value in np.unique(target_weekdays):
                mask = (target_weekdays == day_value)
                model = self.models.get(day_value)
                
                if model is None:
                    raise RuntimeError(f"{day_value}요일 모델이 로드되지 않았습니다.")
                    
                day_X_seq = X_seq[mask]
                day_target_times = target_times[mask]
                
                results.append(self._run_inference(model, day_X_seq, day_target_times, batch_size))
                
        if not results:
            raise RuntimeError("예측 결과가 생성되지 않았습니다.")
            
        final_df = pd.concat(results, ignore_index=True)
        if len(final_df) != len(X_seq):
            raise RuntimeError(
                f"예측 결과가 일부 누락되었습니다: expected={len(X_seq)}, actual={len(final_df)}"
            )
        return final_df.sort_values('datetime').reset_index(drop=True)

    def _run_inference(self, model, X_seq, target_times, batch_size):
        """PyTorch 모델 추론을 수행하는 헬퍼 메소드"""
        X_tensor = torch.FloatTensor(X_seq)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds_list, probs_list = [], []
        for (X_batch,) in loader:
            logits = model(X_batch.to(self.device))
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs[:, 1].cpu().numpy())
            
        return pd.DataFrame({
            'datetime': target_times,
            'predicted_idle_time': preds_list,
            'run_probability': probs_list
        })


def main():
    checkpoint_dir = "checkpoints"
    input_csv = "data/1_month_interference_data.csv"
    output_csv = "results/inference_result.csv"
    
    print(f"데이터 로딩 중: {input_csv}")
    df = load_activity_csv(input_csv, is_inference=True)
    
    print("모델 및 스케일러 로딩 중...")
    predictor = CompressorPredictor(checkpoint_dir=checkpoint_dir, use_union_model=False)
    
    print("예측 수행 중...")
    result_df = predictor.predict(df)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"추론 완료! 총 {len(result_df):,}건의 결과를 저장했습니다.")
    print(f"저장 경로: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
