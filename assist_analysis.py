# --------------------------------------------------------------------------
# assist_analysis.py
#
# 【概要】歩行周期データとLabChart信号を照合し、アシスト開始タイミング(%)を解析する
# 　　　　★修正版v6: config.pyの定義値をフル活用した完全版
#
# --------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config 
import data_processing 

# --------------------------------------------------------------------------

def check_signal_alignment(task_key, labchart_path, output_dir):
    """【デバッグ用】Fzとアシスト信号のタイミングズレ確認"""
    print(f"\n>> 信号のタイミング確認モード: {task_key}")
    try:
        assist_data = data_processing.treadmill_data(labchart_path, config.ASSIST_CHANNEL)
        fz_data = data_processing.treadmill_data(labchart_path, config.Rfz_col)
    except Exception as e:
        print(f"  エラー: データ読み込み失敗 - {e}"); return

    # グラフ表示範囲（configのX軸範囲を使用、なければデフォルト）
    start_time = getattr(config, 'WALK_START_TIME', 120)
    duration = 10 
    fs = 1000 
    s_idx = int(start_time * fs)
    e_idx = int((start_time + duration) * fs)
    if len(assist_data) < e_idx: e_idx = len(assist_data)
    
    time_axis = np.linspace(start_time, start_time + (e_idx-s_idx)/fs, e_idx-s_idx)
    y_assist = assist_data[s_idx:e_idx]
    y_fz = fz_data[s_idx:e_idx]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time_axis, y_fz, color='blue', alpha=0.6, label='Right Fz')
    ax1.set_ylabel('Fz [V]', color='blue'); ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.plot(time_axis, y_assist, color='red', linewidth=2, label='Assist')
    ax2.set_ylabel('Assist [V]', color='red')
    ax2.axhline(config.ASSIST_THRESHOLD, color='green', linestyle='--', label='Th')

    plt.title(f'Check: {task_key} ({start_time}s~)')
    plt.tight_layout()
    
    debug_dir = os.path.join(output_dir, "debug_alignment")
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"Check_Signal_{task_key}.png")
    plt.savefig(save_path)
    print(f"  [Check] 保存: {save_path}")
    plt.show()

def detect_assist_timing(gait_df, assist_signal, sampling_rate=1000):
    """
    歩行周期ごとにアシスト開始タイミングを検出する
    【ロジック】
    1. 前の周期からの「予約」があれば候補Aとする。
    2. 周期内アシスト探索:
       - 開始時間 + config.ASSIST_DURATION_MS > Next_HS なら「次回予約」へ。
       - それ以外なら「候補B (Current)」へ。
    3. 候補Aと候補Bがある場合、実際の信号長(実測値)で比較して採用する。
    """
    start_times, start_frames, start_percents = [], [], []
    threshold = config.ASSIST_THRESHOLD
    
    # ms -> frame 変換 (サンプリングレート依存)
    expected_dur_frames = int(config.ASSIST_DURATION_MS * (sampling_rate / 1000))

    # 次の周期へ渡すデータ
    pending_assist_info = None

    print(f"  解析中: 全{len(gait_df)}周期 (閾値: {threshold}V, 予測継続長: {config.ASSIST_DURATION_MS}ms)")
    
    for index, row in gait_df.iterrows():
        hs_frame = int(row['hs_frame'])
        next_hs_frame = int(row['next_hs_frame'])
        duration = next_hs_frame - hs_frame
        
        if next_hs_frame > len(assist_signal):
            start_times.append(np.nan); start_frames.append(np.nan); start_percents.append(np.nan)
            continue

        # --- 候補管理 ---
        candidate_A = None # 前からの予約分 (Carry-over)
        candidate_B = None # 今回周期内の分 (Current-Cycle)
        next_reservation = None # 次回への予約分 (Reserved)

        # ---------------------------------------------------------
        # 1. 前の周期からの予約を確認 (Candidate A)
        # ---------------------------------------------------------
        if pending_assist_info is not None:
            abs_frame = pending_assist_info['frame']
            abs_time = pending_assist_info['time']
            real_len = pending_assist_info['real_duration']
            
            # パーセント計算 (負の値になる)
            pct = ((abs_frame - hs_frame) / duration) * 100
            
            candidate_A = {
                'pct': pct,
                'frame': abs_frame,
                'time': abs_time,
                'len': real_len
            }

        # ---------------------------------------------------------
        # 2. アシスト検知 & 振り分け
        # ---------------------------------------------------------
        cycle_assist = assist_signal[hs_frame:next_hs_frame]
        is_active = cycle_assist < threshold
        diff = np.diff(is_active.astype(int))
        rising_edges = np.where(diff == 1)[0] + 1 
        falling_edges = np.where(diff == -1)[0] + 1
        
        if is_active[-1]:
            falling_edges = np.append(falling_edges, len(is_active))

        # 新規パルスのループ
        max_len_B = 0
        
        for r_idx in rising_edges:
            if r_idx == 0: continue # 0スタートはCarry-over(Candidate A)で考慮済み
            
            # --- 実測長の計算（比較用） ---
            valid_ends = falling_edges[falling_edges > r_idx]
            if len(valid_ends) == 0:
                 f_idx = duration # 周期端まで
            else:
                 f_idx = valid_ends[0]
            
            current_pulse_len = f_idx - r_idx
            
            # 全体座標
            abs_frame = hs_frame + r_idx
            abs_time = abs_frame / sampling_rate
            
            # --- 【判定】またぎチェック ---
            # Start + config指定時間(200ms) > Next HS ?
            predicted_end_frame = abs_frame + expected_dur_frames
            
            if predicted_end_frame > next_hs_frame:
                # 【次回予約】
                # まだ予約が埋まっていない場合のみ（最初のひとつを優先）
                if next_reservation is None:
                    # 次回比較用に「実際の総信号長」を計算しておく
                    global_start = abs_frame
                    future_signal = assist_signal[global_start:]
                    future_inactive = future_signal >= threshold
                    if np.any(future_inactive):
                        total_len = np.argmax(future_inactive)
                    else:
                        total_len = len(future_signal)

                    next_reservation = {
                        'frame': abs_frame,
                        'time': abs_time,
                        'real_duration': total_len
                    }
            else:
                # 【今回周期データ (Candidate B)】
                # 複数ある場合は、長さが最大のものを候補Bとする
                if current_pulse_len > max_len_B:
                    max_len_B = current_pulse_len
                    start_pct = (r_idx / duration) * 100
                    candidate_B = {
                        'pct': start_pct,
                        'frame': abs_frame,
                        'time': abs_time,
                        'len': current_pulse_len
                    }

        # ---------------------------------------------------------
        # 3. 最終決定 (比較)
        # ---------------------------------------------------------
        winner = None

        if candidate_A is not None and candidate_B is not None:
            # 両方ある -> 長さで比較
            if candidate_A['len'] >= candidate_B['len']:
                winner = candidate_A
            else:
                winner = candidate_B
        elif candidate_A is not None:
            winner = candidate_A
        elif candidate_B is not None:
            winner = candidate_B
        
        # 記録
        if winner is not None:
            start_frames.append(winner['frame'])
            start_times.append(winner['time'])
            start_percents.append(winner['pct'])
        else:
            start_frames.append(np.nan)
            start_times.append(np.nan)
            start_percents.append(np.nan)

        # 次のループへ情報を渡す
        pending_assist_info = next_reservation

    # --- 結果出力用処理 ---
    df_result = gait_df.copy()
    df_result['Assist_Start_Time[s]'] = start_times
    df_result['Assist_Start_Frame'] = start_frames
    df_result['Assist_Start_Percent[%]'] = start_percents
    
    valid_data = [p for p in start_percents if not np.isnan(p)]
    
    if valid_data:
        stats_all = {
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'count': len(valid_data)
        }
    else:
        stats_all = {'mean': np.nan, 'std': np.nan, 'count': 0}

    stats = {
        'all': stats_all,
        'nonzero': stats_all, 
        'total': len(gait_df),
        'nan_count': len(gait_df) - len(valid_data)
    }
    
    return df_result, stats

def create_plots(time_data, percent_data, task_name, leg_label, stats):
    """
    グラフ作成: ここでだけ時間軸を相対時間 (0始まり) に変換してプロット
    """
    valid_data = [(t, p) for t, p in zip(time_data, percent_data) if not np.isnan(t) and not np.isnan(p)]
    if not valid_data: return None, None

    # 絶対時間のリストを取得
    abs_times, percents = zip(*valid_data)
    
    # --- ★時間軸の変換処理 ---
    walk_start = getattr(config, 'WALK_START_TIME', 0)
    walk_end = getattr(config, 'WALK_END_TIME', walk_start + 180)

    # 絶対時間から開始時間を引いて「相対時間」にする
    rel_times = [t - walk_start for t in abs_times]

    mean_val = stats['all']['mean']
    std_val = stats['all']['std']
    
    # グラフのX軸範囲（0 〜 持続時間）
    x_limit_max = walk_end - walk_start

    # --- 1. 時系列グラフ ---
    fig_series, ax_s = plt.subplots(figsize=(10, 6))
    ax_s.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5) 
    
    # プロットには相対時間 (rel_times) を使用
    ax_s.scatter(rel_times, percents, alpha=0.6, c='blue', s=30, label='Assist Timing')
    
    if not np.isnan(mean_val):
        ax_s.axhline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.1f}%')
        ax_s.axhspan(mean_val - std_val, mean_val + std_val, color='red', alpha=0.1)

    ax_s.set_title(f'Assist Timing Series - {leg_label}\n({task_name})')
    ax_s.set_xlabel('Time [s] (Relative to Walk Start)')
    ax_s.set_ylabel('Start Timing [%]')
    
    # X軸を 0スタート に設定
    ax_s.set_xlim(0, x_limit_max)
    
    ax_s.set_ylim(config.ASSIST_PLOT_Y_RANGE) 
    ax_s.legend(loc='upper right')
    ax_s.grid(True, linestyle='--', alpha=0.6)

    # --- 2. ヒストグラム ---
    fig_hist, ax_h = plt.subplots(figsize=(8, 6))
    ax_h.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    
    ax_h.hist(percents, 
              bins=config.ASSIST_HIST_BINS, 
              range=config.ASSIST_HIST_RANGE,
              color='skyblue', edgecolor='black', alpha=0.7)
    
    if not np.isnan(mean_val):
        ax_h.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.1f}%')
    
    ax_h.set_title(f'Assist Timing Dist - {leg_label}\n({task_name})')
    ax_h.set_xlabel('Start Timing [%] (Negative = Pre-HeelStrike)')
    ax_h.set_ylabel('Frequency')
    
    ax_h.set_xlim(config.ASSIST_HIST_RANGE)
    ax_h.legend()
    ax_h.grid(True, axis='y', linestyle='--', alpha=0.5)

    return fig_series, fig_hist

def process_single_leg(task_key, leg_label, csv_filename, labchart_data, output_base_dir):
    """片足分の処理を実行"""
    csv_path = os.path.join(output_base_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"  スキップ: CSVファイルが見つかりません ({leg_label})")
        return

    try: df_cycles = pd.read_csv(csv_path)
    except: return

    # 1. 解析実行
    df_assist, stats = detect_assist_timing(df_cycles, labchart_data)
    
    # 2. 結果表示
    count = stats['all']['count']
    total = stats['total']
    rate = (count / total) * 100 if total > 0 else 0
    
    print(f"\n  [{leg_label}] 検知率: {count}/{total} ({rate:.1f}%)")
    
    if count > 0:
        print(f"  ★ 平均タイミング: {stats['all']['mean']:.2f}% ± {stats['all']['std']:.2f}")
    else:
        print("  ★ 有効アシストなし")
        return

    # 3. CSV保存確認
    while True:
        try:
            print("\n------------------------------------------------")
            save_csv_choice = int(input(f"  [{leg_label}] 解析結果をCSVに保存しますか？ (YES=1, NO=0): "))
            if save_csv_choice in [0, 1]: break
        except ValueError: pass

    assist_dir = os.path.join(output_base_dir, "assist")
    plot_dir = os.path.join(assist_dir, "plots")
    
    if save_csv_choice == 1:
        os.makedirs(assist_dir, exist_ok=True)
        # configの接尾辞を使用
        save_filename = csv_filename.replace('.csv', config.SUFFIX_ASSIST)
        save_path = os.path.join(assist_dir, save_filename)
        
        while True:
            try:
                df_assist.to_csv(save_path, index=False)
                print(f"  ★ CSV保存完了: {save_path}")
                break
            except PermissionError:
                print("\n  🚨【エラー】ファイルが開かれています！")
                input("  >> 閉じてからEnterキーを押してください...")
            except Exception as e:
                print(f"  エラー: {e}")
                break

    # 4. グラフ作成
    fig_series, fig_hist = create_plots(
        df_assist['Assist_Start_Time[s]'], 
        df_assist['Assist_Start_Percent[%]'], 
        task_key, leg_label, stats
    )

    if fig_series is None: return

    # 5. グラフ保存確認
    while True:
        try:
            print("\n------------------------------------------------")
            save_plot_choice = int(input(f"  [{leg_label}] グラフ(時系列・分布)を保存しますか？ (YES=1, NO=0): "))
            if save_plot_choice in [0, 1]: break
        except ValueError: pass

    if save_plot_choice == 1:
        os.makedirs(plot_dir, exist_ok=True)
        path_s = os.path.join(plot_dir, f"Assist_Series_{task_key}_{leg_label}.png")
        path_h = os.path.join(plot_dir, f"Assist_Hist_{task_key}_{leg_label}.png")
        try:
            fig_series.savefig(path_s)
            fig_hist.savefig(path_h)
            print(f"  ★ グラフ保存完了")
        except Exception as e:
            print(f"  エラー: {e}")

    print("  グラフを表示します。閉じる(×)と次に進みます。")
    plt.show()

def main():
    print("--- アシストタイミング解析 (v6: config完全統合版) ---")
    labchart_root = os.path.join(config.BASE_DATA_DIR, config.LABCHART_DIR_NAME)

    while True:
        print("\n================================================")
        print("解析するタスクを選択してください:")
        available_tasks = []
        
        # TASKS_TO_PROCESSの順番で表示しつつ、ASSIST_TASKSに含まれるかチェックも可能
        # ここでは基本設定通り TASKS_TO_PROCESS を使用します
        for i, task_key in enumerate(config.TASKS_TO_PROCESS):
            t_name = config.TASK_TITLES.get(task_key, task_key)
            print(f"  {i+1}: {t_name} ({task_key})")
            available_tasks.append(task_key)
        
        print("  c: 信号確認 (Check)")
        choice = input("番号 (q: 終了): ").lower()
        if choice == 'q': break
        
        if choice == 'c':
            chk = input("確認タスク番号: ")
            try:
                idx = int(chk) - 1
                if 0 <= idx < len(available_tasks):
                    tk = available_tasks[idx]
                    path = os.path.join(labchart_root, config.TASKS[tk])
                    out = os.path.join(labchart_root, tk)
                    if os.path.exists(path): check_signal_alignment(tk, path, out)
            except: pass
            continue

        try:
            t_idx = int(choice) - 1
            if 0 <= t_idx < len(available_tasks):
                target_task = available_tasks[t_idx]
            else: continue
        except: continue

        print(f"\n>> タスク '{target_task}' 開始...")
        txt_path = os.path.join(labchart_root, config.TASKS[target_task])
        if not os.path.exists(txt_path): print("ファイルなし"); continue
            
        print(" LabChart読込中...")
        try: assist_data = data_processing.treadmill_data(txt_path, config.ASSIST_CHANNEL)
        except: print(" 読込失敗"); continue

        task_dir = os.path.join(labchart_root, target_task)
        # process_single_leg(target_task, "Right", f"{target_task}{config.SUFFIX_RIGHT}", assist_data, task_dir)
        process_single_leg(target_task, "Left", f"{target_task}{config.SUFFIX_LEFT}", assist_data, task_dir)
        print("\n  完了。")

if __name__ == "__main__":
    main()