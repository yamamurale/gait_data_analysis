#-----------------------------------------------
# gaitcycle_vector.py
# ベクトル図描画プログラム
#
# --- 処理内容---
# 1. labchart生データ　+ 周期データ「gait_cycle_(right/left).csv」 ( + アシストありタスクのみ「_assist.csv」）　を読み込む
# 2. gait_cycle_force_labchart.py　と同じロジックで 体重正規化 ＋ 周期平均床反力 取得
# 3. ベクトル表示（表示形式ユーザー選択）
# 4. 保存の場合　.....lab chart/task/vector フォルダに保存
# 5. 左右歩行周期　（＋アシスト開始平均）　表示
# 
# ---ユーザー選択内容---
# 1.タスク内容：config.TASKS_TO_PROCESS　から選択
# 2.プロットモード：片足ごとのグラフ（R+L）　or 両足重ね合わせグラフ（ 横軸　Time + % ）
# 3.データ範囲： 1.全データ　→　アシストの有無に関わらず指定範囲内全データを使用。アシスト表記なし
#               2.アシストあり（Non-Zero) → アシストあり（立脚開始時点ですでにアシストONの場合除く）歩行周期のみのデータ使用。アシスト開始平均・標準偏差・ON時間を表示  
#               3.範囲指定（range）→ 入力した範囲内（入力値を含む）にアシストがあった歩行周期のみデータ使用。アシスト開始平均・標準偏差・ON時間を表示
#                            　　　　　　　ex1.) Min %: 40, Max %: 60 入力した場合→　「　40 <= アシストタイミング[%Cycle] <=60%　」に対応した歩行周期のみのデータを使用 
#                            　　　　　　　ex2.) Min %: 0,  Max %: 100 入力した場合→　立脚開始時点ですでにアシストONの場合含めたアシスト開始平均・標準偏差・ON時間を表示
# 4.時間トリミング：config.ANALYSIS_START_TIME ~ config.ANALYSIS_END_TIME (安定期間)　のみに解析を限定するか
#                  (エラー：config.ASK_TRIM_FOR_RAW_CSV = True とし、歩行周期csvの時点で時間を限定しているとエラーが出る可能性があるので注意。False推奨)
#
#-----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import config 
import data_processing

# ==========================================
# 1. データ読み込み & 前処理クラス/関数
# ==========================================

def read_csv_data(filepath):
    """CSV読み込み"""
    if not os.path.exists(filepath): return None
    try: return pd.read_csv(filepath)
    except: return None

def load_task_raw_data(task_key, labchart_dir):
    print(f"  \n生データを読み込み中...")
    
    txt_filename = config.TASKS[task_key]
    txt_path = os.path.join(labchart_dir, txt_filename)
    
    raw_data = {'R': {}, 'L': {}}
    
    try:
        # 1. 全チャンネル読み込み (Fzはドリフト補正付き)
        # Right
        raw_data['R']['Fx'] = data_processing.treadmill_data(txt_path, config.Rfx_col)
        raw_data['R']['Fy'] = data_processing.treadmill_data(txt_path, config.Rfy_col)
        raw_data['R']['Fz'] = data_processing.adjusted_data(txt_path, config.Rfz_col, config.ANALYSIS_START_TIME) 
        
        # Left
        raw_data['L']['Fx'] = data_processing.treadmill_data(txt_path, config.Lfx_col)
        raw_data['L']['Fy'] = data_processing.treadmill_data(txt_path, config.Lfy_col)
        raw_data['L']['Fz'] = data_processing.adjusted_data(txt_path, config.Lfz_col, config.ANALYSIS_START_TIME) 

    except Exception as e:
        print(f"  [エラー] 生データ読み込み失敗: {e}")
        return None, None

    # 2. 体重推定 (読み込んだFzを使用)
    bw = 1.0
    try:
        s_idx = int(config.STAND_START_TIME * 1000)
        e_idx = int(config.STAND_END_TIME * 1000)
        
        # データ長チェック
        if len(raw_data['R']['Fz']) > e_idx:
            rfz_stand = raw_data['R']['Fz'][s_idx:e_idx]
            lfz_stand = raw_data['L']['Fz'][s_idx:e_idx]
            bw = np.mean(rfz_stand) + np.mean(lfz_stand)
            if bw <= 0: bw = 1.0
    except:
        print("  [警告] 体重推定失敗 (1.0Vで代用)")

    print(f"推定：{bw}")

    # 3. Fyドリフト補正 (Config依存)
    if config.DRIFT_CORRECTION:
        start_t = config.ANALYSIS_START_TIME
        # Right
        bias_r = data_processing.calc_static_bias_from_swing(raw_data['R']['Fy'], raw_data['R']['Fz'], start_t, 10.0, "Right_Fy")
        raw_data['R']['Fy'] -= bias_r
        # Left
        bias_l = data_processing.calc_static_bias_from_swing(raw_data['L']['Fy'], raw_data['L']['Fz'], start_t, 10.0, "Left_Fy")
        raw_data['L']['Fy'] -= bias_l

    return raw_data, bw

def filter_dataframe_by_time(df, start_time, end_time, sampling_rate=1000):
    """
    データフレーム内の歩行周期を、指定された時間範囲で絞り込む
    """
    if start_time is None or end_time is None:
        return df
    
    s_frame = int(start_time * sampling_rate)
    e_frame = int(end_time * sampling_rate)
    
    # hs_frame(開始) >= 指定開始  AND  next_hs_frame(終了) <= 指定終了
    # 完全に範囲内に収まっているものだけを抽出
    filtered_df = df[
        (df['hs_frame'] >= s_frame) & 
        (df['next_hs_frame'] <= e_frame)
    ]
    return filtered_df

# ==========================================
# 2. 計算コア
# ==========================================

def find_paired_cycles(left_df, right_df):
    """左足周期に対応する右足周期を見つける"""
    valid_right_rows = []
    time_diffs = []
    phase_diffs = []

    right_df = right_df.sort_values('hs_frame')
    
    for _, l_row in left_df.iterrows():
        l_start = int(l_row['hs_frame'])
        l_end = int(l_row['next_hs_frame'])
        l_dur = l_end - l_start
        
        # 左周期内にある右接地を探す
        candidates = right_df[
            (right_df['hs_frame'] > l_start) & 
            (right_df['hs_frame'] < l_end + (l_dur * 0.2))
        ]
        
        if len(candidates) > 0:
            r_row = candidates.iloc[0]
            valid_right_rows.append(r_row)
            
            diff = int(r_row['hs_frame']) - l_start
            time_diffs.append(diff / 1000.0) 
            phase_diffs.append((diff / l_dur) * 100)

    if not valid_right_rows: return None, None

    paired_right_df = pd.DataFrame(valid_right_rows)
    delays = {'time_sec': np.mean(time_diffs), 'phase_pct': np.mean(phase_diffs)}
    return paired_right_df, delays

def calculate_vector_mean(cycle_df, raw_data_leg, bw):
    """
    指定された周期リストに基づきベクトル平均を計算
    raw_data_leg: 片足分の辞書 {'Fx':..., 'Fy':..., 'Fz':...}
    """
    Fx = raw_data_leg['Fx']
    Fy = raw_data_leg['Fy']
    Fz = raw_data_leg['Fz']
    
    norm_fx, norm_fy, norm_fz = [], [], []
    x_new = np.linspace(0, 100, 101)

    for _, row in cycle_df.iterrows():
        start = int(row['hs_frame'])
        end = int(row['next_hs_frame'])
        
        # 切り出し & BW正規化
        cycle_fx = (Fx[start:end] / bw) * 100
        cycle_fy = (Fy[start:end] / bw) * 100
        cycle_fz = (Fz[start:end] / bw) * 100
        
        # 時間正規化
        x_old = np.linspace(0, 100, len(cycle_fx))
        norm_fx.append(np.interp(x_new, x_old, cycle_fx))
        norm_fy.append(np.interp(x_new, x_old, cycle_fy))
        norm_fz.append(np.interp(x_new, x_old, cycle_fz))

    # 平均歩行周期時間
    durations = (cycle_df['next_hs_frame'] - cycle_df['hs_frame']) / 1000.0
    avg_duration = durations.mean()

    df_result = pd.DataFrame({
        'Gait Cycle [%]': x_new,
        'Time [s]': x_new / 100 * avg_duration,
        'Fx_Mean': np.mean(norm_fx, axis=0),
        'Fy_Mean': np.mean(norm_fy, axis=0),
        'Fz_Mean': np.mean(norm_fz, axis=0)
    })
    
    return df_result, avg_duration

# ==========================================
# 3. 描画関数
# ==========================================
def draw_assist_bar(ax, info, is_time_axis):
    """
    アシスト情報の可視化
    - SHOW_REPEAT_AT_END = True  : 100%付近（右端）にも繰り返しの帯を表示する
    - SHOW_REPEAT_AT_END = False : 本来の位置（マイナス含む）だけを表示する
    """
    # ========== 【ここをいじるだけで変更可能】 ==========
    SHOW_REPEAT_AT_END = False   # True: 右側(100%~)も出す / False: 出さない
    # ====================================================

    start_pct = info['start_mean']
    start_sd_pct = info['start_sd']
    dur_sec = info['duration_sec']
    assist_ms = config.ASSIST_DURATION_MS
    
    if is_time_axis:
        # --- 時間軸 (ms) ---
        start_val = (start_pct / 100.0) * dur_sec * 1000
        sd_val = (start_sd_pct / 100.0) * dur_sec * 1000
        end_val = start_val + assist_ms
        unit = "ms"
        
        ax.axvspan(start_val, end_val, color='red', alpha=0.1, label='Assist')
        
        # 平均線とSD
        ax.axvline(start_val, color='red', linestyle='--', linewidth=1.5, label='Mean')
        bar_y = -5
        ax.errorbar(start_val, bar_y, xerr=sd_val, color='black', capsize=4, 
                    fmt='o', markersize=4, elinewidth=1.5, label='SD')
        
        label_txt = f"Start: {start_val:.1f}{unit}"
        ax.text(start_val, bar_y - 8, label_txt, ha='center', fontsize=9, color='red', fontweight='bold')
        
    else:
        # --- 歩行周期軸 (%) ---
        unit = "%"
        start_val = start_pct
        sd_val = start_sd_pct
        assist_len_pct = (assist_ms / 1000.0 / dur_sec) * 100.0
        end_val = start_val + assist_len_pct

        # 1. 本来の区間を描画（マイナスならマイナスのまま描画
        ax.axvspan(start_val, end_val, color='red', alpha=0.1, label='Assist')

        # 2. 1サイクル後 (+100%) の区間を描画するか判定
        if SHOW_REPEAT_AT_END:
            ax.axvspan(start_val + 100, end_val + 100, color='red', alpha=0.1)

        # 3. 平均線とSD（本来の開始点のみに描画）
        ax.axvline(start_val, color='red', linestyle='--', linewidth=1.5, label='Mean')
        
        # SD (Y軸の下の方に描画)
        bar_y = -5
        ax.errorbar(start_val, bar_y, xerr=sd_val, color='black', capsize=4, 
                    fmt='o', markersize=4, elinewidth=1.5, label='SD')
        
        # ラベル表示
        label_txt = f"Start: {start_val:.1f}{unit}"
        ax.text(start_val, bar_y - 8, label_txt, ha='center', fontsize=9, color='red', fontweight='bold')
'''
def draw_assist_bar(ax, info, is_time_axis):
    """
    アシスト情報の可視化 (背景色変更 + 平均線 + SD矢印)
    """
    start_pct = info['start_mean']
    start_sd_pct = info['start_sd']
    dur_sec = info['duration_sec']
    
    # 単位変換
    if is_time_axis:
        # % -> ms
        start_val = (start_pct / 100.0) * dur_sec * 1000
        sd_val = (start_sd_pct / 100.0) * dur_sec * 1000
        assist_len = config.ASSIST_DURATION_MS
        unit = "ms"
    else:
        # %
        start_val = start_pct
        sd_val = start_sd_pct
        assist_len = (config.ASSIST_DURATION_MS / 1000.0 / dur_sec) * 100.0
        unit = "%"
        
    end_val = start_val + assist_len
    
    # 1. 背景色の変更 (Assist Duration)
    # y軸の全範囲にわたって帯を引く
    ax.axvspan(start_val, end_val, color='red', alpha=0.1, label='Assist')
    
    # 2. 開始タイミング (平均) の縦線
    ax.axvline(start_val, color='red', linestyle='--', linewidth=1.5, label='Mean')
    
    # 3. 標準偏差 (SD) の矢印 (← →)
    # Y軸の下の方(-5くらい)に描画
    bar_y = -5
    ax.errorbar(start_val, bar_y, xerr=sd_val, color='black', capsize=4, 
                fmt='o', markersize=4, elinewidth=1.5, label='SD')
    
    # ラベル
    label_txt = f"Start: {start_val:.1f}{unit}"
    ax.text(start_val, bar_y - 8, label_txt, ha='center', fontsize=9, color='red', fontweight='bold')
'''
def save_plot(fig, base_dir, filename):
    """保存・表示共通処理"""
    print(f"\nグラフ生成: {filename}")
    
    vector_dir = os.path.join(base_dir, "vector")
    os.makedirs(vector_dir, exist_ok=True)
    save_path = os.path.join(vector_dir, filename)

    # ユーザーに聞く
    plt.show(block=False)
    while True:
        try:
            c = int(input("  保存しますか？ (YES=1, NO=0): "))
            if c in [0, 1]: break
        except: pass
    
    if c == 1:
        try: fig.savefig(save_path); print(f"  ★ 保存完了: {save_path}")
        except: print("  保存失敗")
    plt.close()

def plot_vector_single(df, save_base_dir, file_basename, leg_label, plane_mode, view_mode, assist_info=None):
    """片足のみプロット"""
    x_col, y_col, x_lab, y_lab = config.MODE_TO_AXES[plane_mode]
    col_x, col_y = f"{x_col}_Mean", f"{y_col}_Mean"
    
    # データ抽出
    idx = np.arange(0, len(df), config.VECTOR_STEP)
    df_sub = df.iloc[idx]
    
    x_vals = df_sub['Gait Cycle [%]'].to_numpy()
    y_zeros = np.zeros_like(x_vals)
    dx = df_sub[col_x].to_numpy()
    dy = df_sub[col_y].to_numpy()

    # 符号反転
    if x_col == 'Fx' and config.FLIP_FX: dx = -dx
    if x_col == 'Fy' and config.FLIP_FY: dx = -dx
    if y_col == 'Fz' and config.FLIP_FZ: dy = -dy
    if y_col == 'Fy' and config.FLIP_FY and plane_mode == 2: dy = -dy 

    mag = np.sqrt(dx**2 + dy**2)

    # 描画
    fig, ax = plt.subplots(figsize=config.VECTOR_FIGSIZE)
    ax.set_aspect('auto')
    
    # angles='uv' (ベクトルの形を成分比で決定), scale_units='y' (長さをY軸基準に)
    q = ax.quiver(x_vals, y_zeros, dx, dy, mag, angles='uv', scale_units='y', scale=1, cmap='viridis', width=0.005, headwidth=4, zorder=3)
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label('Magnitude [%BW]')
    
    ax.set_xlabel('Gait Cycle [%]')
    ax.set_ylabel(f'{y_lab} [%BW]')
    
    task_title = assist_info.get('title', file_basename) if assist_info else file_basename
    suffix = ["", "", " (Assist Non-Zero)", " (Assist Range)"][view_mode] if view_mode < 4 else ""
    
    plane_name = f"{x_col}-{y_col}"
    ax.set_title(f'{task_title} - {leg_label}', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(config.XLIM)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    
    if plane_mode in [0, 1]: ax.set_ylim(config.FZ_YLIM)
    else: ax.set_ylim(config.FY_YLIM)
    ax.axhline(0, color='black', linewidth=1) 

    if assist_info and not np.isnan(assist_info.get('start_mean', np.nan)):
        draw_assist_bar(ax, assist_info, is_time_axis=False)
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    
    # モードタグ
    mode_tag = ["", "", "_nz", "_range"][view_mode] if view_mode < 4 else ""
    save_name = f"v_{plane_name}_{file_basename}_{leg_label.replace(' ', '')}{mode_tag}.png"
    
    save_plot(fig, save_base_dir, save_name)

def plot_vector_combined(df_l, df_r, delays, save_base_dir, file_basename, plane_mode, view_mode, assist_info=None):
    """両足重ね合わせプロット"""
    x_col, y_col, _, y_lab = config.MODE_TO_AXES[plane_mode]
    col_x, col_y = f"{x_col}_Mean", f"{y_col}_Mean"
    
    idx = np.arange(0, 101, config.VECTOR_STEP)
    
    l_x_pct = df_l['Gait Cycle [%]'].iloc[idx].to_numpy()
    l_x_time = df_l['Time [s]'].iloc[idx].to_numpy() * 1000
    l_dx = df_l[col_x].iloc[idx].to_numpy()
    l_dy = df_l[col_y].iloc[idx].to_numpy()
    
    r_x_pct = df_r['Gait Cycle [%]'].iloc[idx].to_numpy()
    r_x_time = df_r['Time [s]'].iloc[idx].to_numpy() * 1000
    r_dx = df_r[col_x].iloc[idx].to_numpy()
    r_dy = df_r[col_y].iloc[idx].to_numpy()

    if x_col == 'Fx' and config.FLIP_FX: l_dx = -l_dx; r_dx = -r_dx
    if x_col == 'Fy' and config.FLIP_FY: l_dx = -l_dx; r_dx = -r_dx
    if y_col == 'Fz' and config.FLIP_FZ: l_dy = -l_dy; r_dy = -r_dy
    if y_col == 'Fy' and config.FLIP_FY and plane_mode == 2: l_dy = -l_dy; r_dy = -r_dy

    # カラーマップ「底上げ」→　値の小さいデータが見にくいため
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    # 【設定】色の濃さ調整
    # 例：0.0(白) ～ 1.0(濃) のうち、0.2(淡い色) ～ 1.0(濃い色) だけを使う
    cmap_l = truncate_colormap(plt.get_cmap('Blues'), 0.2, 1.0)
    cmap_r = truncate_colormap(plt.get_cmap('Oranges'), 0.2, 1.0)

    def draw_combined(ax, x_l, x_r, x_label, x_limit, is_time):
        
        ax.set_aspect('auto')
        zeros = np.zeros_like(x_l)

        mag_l = np.sqrt(l_dx**2 + l_dy**2)
        ax.quiver(x_l, zeros, l_dx, l_dy, mag_l, angles='uv', scale_units='y', scale=1, 
                  cmap=cmap_l, width=0.005, headwidth=4, clim=(0, 150), 
                   linewidth=0.2, zorder=3)
        mag_r = np.sqrt(r_dx**2 + r_dy**2)
        ax.quiver(x_r, zeros, r_dx, r_dy, mag_r, angles='uv', scale_units='y', scale=1, 
                  cmap=cmap_r, width=0.005, headwidth=4, clim=(0, 150), linewidth=0.2, zorder=3)

        ax.set_xlabel(x_label)
        ax.set_ylabel(f'{y_lab} [%BW]')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(x_limit)
        
        if plane_mode in [0, 1]: ax.set_ylim(config.FZ_YLIM)
        else: ax.set_ylim(config.FY_YLIM)
        ax.axhline(0, color='black', linewidth=1)
        
        ax.plot([], [], color='blue', label='Left')
        ax.plot([], [], color='orange', label='Right')
        ax.legend(loc='upper right')

    task_title = assist_info.get('title', file_basename) if assist_info else file_basename
    plane_name = f"{x_col}-{y_col}"
    mode_tag = ["", "", "_nz", "_range"][view_mode] if view_mode < 4 else ""

    # 1. 周期軸
    fig1, ax1 = plt.subplots(figsize=config.COMBINED_FIGSIZE) 
    r_x_shifted_pct = r_x_pct + delays['phase_pct']
    draw_combined(ax1, l_x_pct, r_x_shifted_pct, 'Gait Cycle [%]', config.XLIM_COMBINED_PCT, False)
    if assist_info and not np.isnan(assist_info.get('start_mean', np.nan)):
        draw_assist_bar(ax1, assist_info, False)
    ax1.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
    ax1.legend(loc='upper right', fontsize='small', framealpha=0.9)
    ax1.set_title(f'{task_title}')
    plt.tight_layout()
    save_plot(fig1, save_base_dir, f"v_both_{plane_name}_{file_basename}{mode_tag}.png")

    # 2. 時間軸
    fig2, ax2 = plt.subplots(figsize=config.COMBINED_FIGSIZE)
    delay_ms = delays['time_sec'] * 1000
    r_x_shifted_time = l_x_time + delay_ms 
    xlim_ms = (config.XLIM_COMBINED_TIME[0]*1000, config.XLIM_COMBINED_TIME[1]*1000)
    
    draw_combined(ax2, l_x_time, r_x_shifted_time, 'Time [ms]', xlim_ms, True)
    if assist_info and not np.isnan(assist_info.get('start_mean', np.nan)):
        draw_assist_bar(ax2, assist_info, True)
    ax2.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])
    ax2.legend(loc='upper right', fontsize='small', framealpha=0.9)
    ax2.set_title(f'{task_title}')
    plt.tight_layout()
    save_plot(fig2, save_base_dir, f"v_both_time_{plane_name}_{file_basename}{mode_tag}.png")


# ==========================================
# 4. メインループ
# ==========================================

def main():
    print("--- ベクトル図作成 ---")
    labchart_dir = os.path.join(config.BASE_DATA_DIR, config.LABCHART_DIR_NAME)
    
    while True:
        print("\n================================================")
        # 1. タスク選択
        print("処理するタスクを選択してください:")
        available_tasks = []
        for i, task_key in enumerate(config.TASKS_TO_PROCESS):
            t_name = config.TASK_TITLES.get(task_key, task_key)
            print(f"  {i+1}: {t_name} ({task_key})")
            available_tasks.append(task_key)
        
        choice = input("番号 (q: 終了): ")
        if choice.lower() == 'q': break
        try:
            task_idx = int(choice) - 1
            if not (0 <= task_idx < len(available_tasks)): continue
            target_task = available_tasks[task_idx]
        except: continue

        # 2. モード選択 (片足 or 両足)
        print("\nプロットモード:")
        print("  1: 片足ずつ ")
        print("  2: 両足重ね合わせ ")
        try:
            plot_mode = int(input("選択 (1-2): "))
            if plot_mode not in [1, 2]: continue
        except: continue

        # 3. フィルタリング選択
        has_assist = (target_task in config.ASSIST_TASKS)
        print("\nデータの範囲:")
        print("  1: 全データ(アシスト表記なし)")
        if has_assist:
            print("  2: アシストあり ")
            print("  3: 範囲指定 (Range)")
        
        try:
            view_mode = int(input("選択: "))
            if not (view_mode == 1 or (has_assist and view_mode in [2, 3])): continue
        except: continue

        # 4. 平面選択 (デフォルト Fy_Fz ※変更したい場合コメントアウトしている部分を復活させてください)
        plane = 0
        '''
        print("\n表示する平面:")
        print("  0: Fy - Fz"); print("  1: Fx - Fz"); print("  2: Fx - Fy")
        try:
            plane = int(input("選択 (0-2): "))
            if plane not in [0, 1, 2]: continue
        except: continue
        '''
        # === 処理開始 ===
        # A. 生データと体重を読み込む 
        raw_data, bw = load_task_raw_data(target_task, labchart_dir)
        if raw_data is None: continue

        task_dir = os.path.join(labchart_dir, target_task)

        # B. CSVファイルの特定 
        suffix_l = config.SUFFIX_LEFT
        csv_l_path = os.path.join(task_dir, f"{target_task}{suffix_l}")
        assist_l_path = os.path.join(task_dir, "assist", f"{target_task}{suffix_l.replace('.csv', config.SUFFIX_ASSIST)}")
        
        # 読み込むCSV 
        target_csv_path_l = assist_l_path if (has_assist and os.path.exists(assist_l_path)) else csv_l_path
        if not os.path.exists(target_csv_path_l): print("左足CSVなし"); continue
        df_l_all = read_csv_data(target_csv_path_l)

        # 右足CSV (全データ用)
        csv_r_path = os.path.join(task_dir, f"{target_task}{config.SUFFIX_RIGHT}")
        df_r_all = read_csv_data(csv_r_path)
        
        # 時間トリミングの質問と実行
        start_t = config.ANALYSIS_START_TIME
        end_t = config.ANALYSIS_END_TIME
        use_time_trim = False
        
        while True:
            try:
                print("\n------------------------------------------------")
                choice = int(input(f"[時間設定] 解析に {start_t}秒～{end_t}秒 の期間のみを使用しますか？ (YES=1, NO=0 [全期間]): "))
                if choice == 1: use_time_trim = True; break
                elif choice == 0: use_time_trim = False; break
            except ValueError: pass

        if use_time_trim:
            # 左足をトリミング
            df_l_all = filter_dataframe_by_time(df_l_all, start_t, end_t)
            # 右足もトリミング 
            if df_r_all is not None:
                df_r_all = filter_dataframe_by_time(df_r_all, start_t, end_t)
            print(f"  -> 時間トリミング後(Left): {len(df_l_all)} 歩")

        if len(df_l_all) == 0:
            print("  エラー: 指定期間内のデータがありません。"); continue
        
        # C. フィルタリング (左足基準)
        df_l_filtered = df_l_all
        if view_mode == 2:
            df_l_filtered = df_l_all[df_l_all['Assist_Start_Percent[%]'].notna()]
        elif view_mode == 3:
            try:
                min_p = float(input("Min %: "))
                max_p = float(input("Max %: "))
                df_l_filtered = df_l_all[(df_l_all['Assist_Start_Percent[%]'] >= min_p) & 
                                         (df_l_all['Assist_Start_Percent[%]'] <= max_p)]
            except: df_l_filtered = df_l_all
        
        print(f"  -> 対象周期 (Left): {len(df_l_filtered)} 歩")

        # 「左足に対応する右足」だけを使う
        df_r_filtered = df_r_all
        delays = None
        
        if df_r_all is not None:
            # 常にペアリングを実行して、対応する右足を抽出する
            df_r_paired, delays = find_paired_cycles(df_l_filtered, df_r_all)
            if df_r_paired is not None:
                df_r_filtered = df_r_paired
                print(f"  -> 対象周期(Right): {len(df_r_filtered)} 歩 (ペアリング済み)")
            else:
                print("  [警告] 右足ペアリング失敗。全データを使用します。")
        
        # D.フィルタリング後の歩行周期[s]＋標準偏差　表示
        l_durs = (df_l_filtered['next_hs_time'] - df_l_filtered['hs_time'])
        l_avg_dur = l_durs.mean()
        l_std_dur = l_durs.std()
        
        r_avg_dur = 0
        r_std_dur = 0

        if df_r_filtered is not None:
            r_durs = (df_r_filtered['next_hs_time'] - df_r_filtered['hs_time'])
            r_avg_dur = r_durs.mean()
            r_std_dur = r_durs.std()
            
        print("\n================================================")
        print(f"  【歩行周期時間 (Mean ± SD)】")
        print(f"    Left : {l_avg_dur:.4f} ± {l_std_dur:.4f} s")
        if df_r_filtered is not None:
            print(f"    Right: {r_avg_dur:.4f} ± {r_std_dur:.4f} s")

        # E. アシスト情報作成
        assist_info = None
        task_title = config.TASK_TITLES.get(target_task, target_task)
        if has_assist and view_mode != 1:
            vals = df_l_filtered['Assist_Start_Percent[%]']
            assist_info = {'start_mean': vals.mean(), 'start_sd': vals.std(), 'duration_sec': l_avg_dur, 'title': task_title}
            print(f"  【アシスト開始 (Mean ± SD)】")
            print(f"    {assist_info ['start_mean']:.2f} ± {assist_info['start_sd']:.2f} %  ({len(df_l_filtered)}/{len(df_l_all)})")
        else:
            assist_info = {'title': task_title, 'start_mean': np.nan}
        print("================================================")

        # === 計算と描画 ===
        
        if plot_mode == 1:
            # 1. Right (フィルタ済み)
            vec_r, _ = calculate_vector_mean(df_r_filtered, raw_data['R'], bw)
            print("  [Right] 描画中...")
            plot_vector_single(vec_r, task_dir, target_task, "Right Foot", plane, view_mode, assist_info)         
            # 2. Left (フィルタ済み)
            vec_l, _ = calculate_vector_mean(df_l_filtered, raw_data['L'], bw)
            print("  [Left] 描画中...")
            plot_vector_single(vec_l, task_dir, target_task, "Left Foot", plane, view_mode, assist_info)

        else:
            # Combined
            if df_r_filtered is None: continue
            
            # ベクトル計算
            vec_l, _ = calculate_vector_mean(df_l_filtered, raw_data['L'], bw)
            vec_r, _ = calculate_vector_mean(df_r_filtered, raw_data['R'], bw)
            
            print("\n[Combined] 描画中...")
            plot_vector_combined(vec_l, vec_r, delays, task_dir, target_task, plane, view_mode, assist_info)

if __name__ == "__main__":
    main()