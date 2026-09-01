import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager as fm

def setup_font():
    try:
        fm.fontManager.addfont("./font/NanumGothic.ttf")
        plt.rc("font", family="NanumGothic")
    except Exception:
        print("한글 폰트를 찾을 수 없습니다. 폰트 깨짐이 발생할 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

def main():
    setup_font()
    
    points_per_week = 10080 # 1주 단위로 시각화
    
    input_csv = "data/1_month_interference_data.csv"      
    result_csv = "results/inference_result.csv" 
    save_dir = "results"  # 디렉토리

    print("데이터를 불러오는 중...")
    df_in = pd.read_csv(input_csv, parse_dates=['datetime'])
    df_res = pd.read_csv(result_csv, parse_dates=['datetime'])

    # 원본 데이터와 예측 결과 병합
    df = pd.merge(df_in, df_res, on='datetime', how='inner')
    
    if df.empty:
        print("병합된 데이터가 없습니다. 날짜 형식을 확인해주세요.")
        return

    # 1주일 간격으로 확인하는 그래프
    for week in range(1, 6):
        start_idx = (week - 1) * points_per_week
        end_idx = week * points_per_week
        
        # 해당 주차의 데이터 자르기
        plot_df = df.iloc[start_idx:end_idx]
        
        # 남은 데이터가 없으면 종료
        if plot_df.empty:
            print(f"\n{week}주차 데이터가 존재하지 않아 시각화를 종료합니다.")
            break

        # 이번 주차의 저장 파일 경로 생성
        current_save_path = os.path.join(save_dir, f"inference_plot_week{week}.png")
        print(f"\n{week}주차 그래프 생성 중... (데이터 수: {len(plot_df):,}개)")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        color_power = '#1f77b4'
        ax1.plot(plot_df['datetime'], plot_df['power_usage'], color=color_power, label='전력 사용량', linewidth=1)
        ax1.set_ylabel('전력 사용량', color=color_power)
        ax1.tick_params(axis='y', labelcolor=color_power) # 전력 사용량 그래프 파란색
    
        ax1_twin = ax1.twinx()
        color_flow = '#ff7f0e'
        ax1_twin.plot(plot_df['datetime'], plot_df['flow_usage'], color=color_flow, label='유량', linewidth=1, alpha=0.7)
        ax1_twin.set_ylabel('유량', color=color_flow)
        ax1_twin.tick_params(axis='y', labelcolor=color_flow) # 유량 그래프 주황색
    
        ax1.set_title(f"공기압축기 센서 데이터 및 상태 예측 결과 - {week}주차", fontsize=16, pad=15)
        ax1.grid(True, alpha=0.3)

        color_prob = '#2ca02c'  # 초록색 선 (확률)
        color_state = '#d62728' # 빨간색 점선 (최종 예측 상태)

        # 작동 확률
        ax2.plot(plot_df['datetime'], plot_df['run_probability'], color=color_prob, label='작동 확률 (0~1)', linewidth=1.5)
    
        # 예측된 상태
        ax2.step(plot_df['datetime'], plot_df['predicted_idle_time'], color=color_state, where='post', 
             label='최종 예측 상태 (0:휴지, 1:작동)', linestyle='--', alpha=0.8)
    
        # 0.5 나오는 부분
        ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.6)

        ax2.set_ylabel("상태 / 확률")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.set_yticklabels(["0 (휴지)", "0.5", "1 (작동)"])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # X축 시간 간격
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=30)

        plt.tight_layout()
        plt.savefig(current_save_path)
        plt.close(fig)
        
        print(f"시각화 완료! 이미지가 저장되었습니다: {current_save_path}")
    
if __name__ == "__main__":
    main()