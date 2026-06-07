#data_processing.py (ver2)
# 歩行周期切り出し用のモジュール

import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.signal import butter, filtfilt
import config
import os

__all__ = ['process_all_files', 'calculate_gait_cycles']


#Rfz, Lfz, Btime, Kinema = 3, 10, 15, 16
# --- ▼▼▼ ここから追加 ▼▼▼ ---
# グローバル変数として列番号を定義
# ★列番号の定義をシンプルに（今回はLfzのみ使用）
Lfx_col, Lfy_col, Lfz_col = 1, 2, 3
# --- ▲▲▲ ここまで追加 ▲▲▲ ---

def get_txt_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.txt')]
'''
def process_all_files(directory, based_or_not):
    files = get_txt_files(directory)
    results = {}
    for tn, file in enumerate(files):
        file_path = os.path.join(directory, file)
        print(f"処理中のファイル: {file_path}")
        data = process_data(file_path)
        results[tn] = {
            'Rfz': all_data(file_path, Rfz, based_or_not),
            'Lfz': all_data(file_path, Lfz, based_or_not),
            'Btime': all_data(file_path, Btime, based_or_not),
            'Kinema': all_data(file_path, Kinema, based_or_not)
        }
        print(f"ファイル {file} の処理結果:")
        for key, value in results[tn].items():
            print(f"{key}: {'データあり' if value is not None else 'データなし'}")
    return results
'''

def process_data(file):
    encodings = ['utf-8', 'shift-jis', 'latin-1']
    for encoding in encodings:
        try:
            data = pd.read_csv(file, skiprows=6, header=None, sep='\t', encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the file with the specified encodings")

def interpolate_nan(col):
    mask = col.notna()
    x = np.flatnonzero(mask)
    y = col[mask]
    if len(x) < 4 or len(x) == len(col):  # NaNがない場合や補間に十分なデータがない場合
        return col
    interp = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return pd.Series(interp(np.arange(len(col))), index=col.index)

def lowpass_filter(data, cutoff, fs=1000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def treadmill_data(file, cn, cutoff_frequency=6):#13=sotsuron
    data = process_data(file)
    if cn >= data.shape[1]:
        raise ValueError(f"Column index {cn} is out of bounds. DataFrame has {data.shape[1]} columns.")
    data = data.iloc[:, cn]  # DataFrameの列を選択
    return lowpass_filter(data, cutoff_frequency)

# 追加：Fyドリフト解決　
def calc_static_bias_from_swing(target_data, fz_data, start_time, duration=10.0, label=""):
    """解析開始から10秒間の遊脚期中央値をバイアスとして算出"""
    if start_time is None: start_time = 0
    s_idx = int(start_time * 1000)
    e_idx = int((start_time + duration) * 1000)
    
    if len(fz_data) < e_idx:
        e_idx = len(fz_data)
        if s_idx >= e_idx: return 0.0

    ref_fz = fz_data[s_idx:e_idx]
    ref_target = target_data[s_idx:e_idx]
    threshold = np.max(ref_fz) * 0.05
    swing_mask = ref_fz < threshold
    swing_values = ref_target[swing_mask]
    
    if len(swing_values) == 0: return 0.0
    
    bias = np.median(swing_values)
    print(f"  [静的補正 {label}] {bias:.4f} を補正します。")
    return bias


def adjusted_data(file, cn, start_time):
    filtered_data = treadmill_data(file, cn)
    sample_data = filtered_data[int(start_time*1000):int(start_time*1000)+5000]
    #sample_data = filtered_data[20000:25000]
    
    # 下位10%のデータを抽出
    lower_10_percent = np.percentile(sample_data, 10)
    baseline_data = sample_data[sample_data <= lower_10_percent] 
    # 下位10%の平均を計算
    baseline = np.mean(baseline_data) 

    #if baseline < 0:
    #    baseline = 0.0000001
    
    print(f"adjusted_data関数: cn = {cn}, baseline = {baseline}")

    return filtered_data - baseline

'''
#改良
def adjusted_data(file, cn):
    filtered_data = treadmill_data(file, cn)
    sample_data = filtered_data[20000:25000]
    #20-25秒
    # 下位10%のデータを抽出
    lower_10_percent = np.percentile(sample_data, 10)
    baseline_data = sample_data[sample_data <= lower_10_percent] 
    # 下位10%の平均を計算
    baseline = np.mean(baseline_data) 
    print(f"adjusted_data関数: cn = {cn}, baseline = {baseline}")
    return filtered_data - baseline
'''

'''
def all_data(file, cn, based_or_not):
    print(f"all_data関数: cn = {cn}, based_or_not = {based_or_not}")
    
    if cn in [Rfz, Lfz]:
        if based_or_not == 1:
            result = treadmill_data(file, cn)
        else:
            result = adjusted_data(file, cn)
    elif cn in [Btime, Kinema]:
        data = process_data(file)
        result = data.iloc[:, cn].to_numpy()  # Pandas SeriesをNumPy arrayに変換
    else:
        result = None
        print("無効なcn値")
    
    if result is not None:
        print(f"処理結果（最初の10要素）:")
        print(result[:10])
        print(f"all_data関数の戻り値の長さ: {len(result)}")
    else:
        print("all_data関数の戻り値: None")
    
    return result
'''

def all_data(file, cn, based_or_not):
    if based_or_not == 2: # adjusted_dataを呼び出す場合
        result = adjusted_data(file, cn)
    else: # 元データをフィルターだけかける場合
        result = treadmill_data(file, cn)
    return result


def calculate_gait_cycles_senior(data, force_threshold=0.09, fs=1000.0, offset_frames=-50):
    """
    CYCLE_START_OFFSET_FRAMES (-50) のシフト処理を追加反映
    """
    if len(data) == 0:
        print("警告: 空のデータセットです。")
        return []

    # Numpy配列として扱う
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data)

    diff_data = np.diff(data)
    window_size = 5
    
    # 先輩の設定値（秒）
    MIN_CYCLE_DURATION = 0.4
    MAX_CYCLE_DURATION = 3.0

    stance_starts = []
    stance_ends = []
    n_samples = len(data)

    # Heel Strike (立脚開始) Detection
    for i in range(len(diff_data) - 1):
        if data[i] < force_threshold and data[i+1] > force_threshold:
            if i >= window_size:
                y_reg = data[i - window_size : i]
                x_reg = np.arange(i - window_size, i)
                slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                
                if abs(slope) > 1e-9:
                    actual_start = int(-intercept / slope)
                    if 0 <= actual_start < n_samples:
                        stance_starts.append(actual_start)

    # Toe Off (立脚終了) Detection
    for i in range(len(diff_data) - 1):
        if data[i] > force_threshold and data[i+1] < force_threshold:
            if i + window_size < n_samples:
                y_reg = data[i - window_size : i]
                x_reg = np.arange(i - window_size, i)
                slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                
                if abs(slope) > 1e-9:
                    actual_end = int(-intercept / slope)
                    if 0 <= actual_end < n_samples:
                        stance_ends.append(actual_end)

    gait_cycles = []
    time_offset_sec = offset_frames / fs

    # 歩行周期のペアリング
    for start_idx in stance_starts:
        start_time = start_idx / fs
        
        next_starts = [s for s in stance_starts if s > start_idx]
        if next_starts:
            next_start_idx = min(next_starts)
            next_start_time = next_start_idx / fs
            cycle_duration = next_start_time - start_time
            
            if MIN_CYCLE_DURATION < cycle_duration < MAX_CYCLE_DURATION:
                valid_ends = [end for end in stance_ends if start_idx < end < next_start_idx]
                if valid_ends:
                    end_idx = min(valid_ends)
                    end_time = end_idx / fs
                    
                    # ----------------------------------------------------
                    # ★重要: 先輩のコードに合わせて hs_frame と hs_time だけシフトする
                    # ----------------------------------------------------
                    shifted_start_idx = int(start_idx + offset_frames)
                    shifted_start_time = start_time + time_offset_sec
                    shifted_next_start_idx = int(next_start_idx + offset_frames)
                    shifted_next_start_time = next_start_time + time_offset_sec
                    
                    gait_cycles.append({
                        'hs_time': shifted_start_time,
                        'to_time': end_time,  # 先輩のコードでは TO はシフトしていない
                        'next_hs_time': shifted_next_start_time,
                        'hs_frame': shifted_start_idx,
                        'to_frame': end_idx,
                        'next_hs_frame': shifted_next_start_idx,
                        'duration': cycle_duration
                    })

    print(f"\n[ロジック適用(50フレームシフト済)] 検出数: {len(gait_cycles)}\n")
    return gait_cycles

def calculate_gait_cycles_ver2(data):
    if len(data) == 0:
        print("警告: 空のデータセットです。")
        return []
    
    # 微分値の計算
    diff_data = np.diff(data)
    
    # 閾値の設定
    rise_threshold = 0.0001   # 立ち上がりの閾値
    fall_threshold = -0.0001  # 降下の閾値
    force_threshold = 0.09   # 力データの閾値
    max_cycle_duration = 1800  # 最大歩行周期時間（2秒 = 2000サンプル）
    
    s_range = 0.5 # 立脚補正に使用する範囲[s]
    p_range = 0.4 # 立脚補正に使用する範囲[s]
    stable_max = 1 # 収束確認の最大繰り返し回数
    stable_len = 20 # 収束確認のウィンドウ長 (10サンプル)
    stable_threshold = 0.008 # ゼロ付近収束の閾値 (V)

    stance_starts = []
    stance_ends = []
    
    # 立脚開始点の検出と補正
    for i in range(len(diff_data)-1):
        
        if (diff_data[i] > rise_threshold and
            data[i] < force_threshold and
            data[i+1] > force_threshold):

            rise_index = i

            # 探索範囲
            search_range = int(s_range * 1000)  
            start_search = max(0, rise_index - search_range)
            end_search = rise_index
            # 上昇直前最小値
            local_min_rel = np.argmin(data[start_search:end_search])
            valley_idx = start_search + local_min_rel

            # 上昇確認
            post_window = int(p_range * 1000)
            if valley_idx + post_window < len(data):
                if np.all(np.diff(data[valley_idx:valley_idx+post_window]) > 0):
                    stance_starts.append(valley_idx)
                else:
                    reg_window = 5
                    if rise_index >= reg_window:
                        y_reg = data[rise_index - reg_window : rise_index]
                        x_reg = np.arange(rise_index - reg_window, rise_index)
                        
                        slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                        
                        # 傾きが正常（右肩上がり）なら 0点を計算
                        if slope > 1e-6: # ほぼ0での除算防止
                            calc_start = int(-intercept / slope)
                            stance_starts.append(calc_start)
                        else:
                            stance_starts.append(rise_index)
                    else:
                        stance_starts.append(rise_index)
    
    # 立脚終了点の検出と補正
    for i in range(len(diff_data)-1):
        
        if (diff_data[i] < fall_threshold and 
            data[i] > force_threshold and 
            data[i+1] < force_threshold):
            
            fall_index = i

            # 最小値探索 (離地検出点の後方で探索)
            search_range_samples = int(s_range * 1000)
            start_search = fall_index
            end_search = min(fall_index + search_range_samples, len(data))
            
            # 探索範囲が有効かチェック
            if start_search >= end_search:
                continue

            local_min_rel = np.argmin(data[start_search:end_search])
            valley_idx = start_search + local_min_rel

            # ゼロ付近収束確認
            stable_count = 0
            j = valley_idx
            max_forward = stable_max * stable_len 

            while j + stable_len < len(data) and stable_count < max_forward:
                local_window = np.abs(np.diff(data[j:j+stable_len]))
                if np.all(local_window < stable_threshold):
                    j += 1
                    stable_count += 1
                else:
                    break

            end_idx = j if stable_count > 0 else valley_idx
            stance_ends.append(end_idx) # timeではなくframe番号を格納
    
   # 歩行周期のペアリング
    gait_cycles = []
    
    for start in stance_starts:
        # 現在の開始点以降で最も近い終了点を探す
        valid_ends = [end for end in stance_ends 
                     if end > start and 
                     end - start < max_cycle_duration]
        
        if valid_ends:
            end = min(valid_ends)  # 最も近い終了点を選択
            # 次の接地を探す
            next_starts = [s for s in stance_starts if s > end]
            if next_starts:
                next_start = min(next_starts)
                gait_cycles.append({
                    'hs_time': start / 1000.0,      # 接地時間 (秒)
                    'to_time': end / 1000.0,        # 離地時間 (秒)
                    'next_hs_time': next_start / 1000.0, # 次の接地時間 (秒)
                    'hs_frame': start,
                    'to_frame': end,
                    'next_hs_frame': next_start
                })
    # 検出結果の出力
    print(f"\n検出された歩行周期の総数: {len(gait_cycles)}\n")
    
    return gait_cycles    



def calculate_gait_cycles_ver4(data):
    """
　　Zero-Base Regression (ゼロベース回帰法)
    """
    if len(data) == 0:
        print("警告: 空のデータセットです。")
        return []
    
    # 1. イベント検知用
    threshold_Force = 0.1     # [V] 接地判定 (これを超えたら計算開始)
    threshold_SearchLimit = 0.03 # [V] これを下回る点までは最低限遡る (探索の限界点)
    
    # 2. 回帰直線の範囲 (直線の傾きを決める)
    reg_voltage_start = 0.01  # 回帰の始点
    reg_voltage_end   = 0.10  # 回帰の終点
    
    # 3. その他
    search_window = 400
    max_cycle_limit = 2000 
    
    # ==========================================
    
    stance_starts = []
    stance_ends = []
    
    n_samples = len(data)
    in_stance = False
    
    print(f"  [Gait Detect] RegRange: {reg_voltage_start}-{reg_voltage_end}V, Zero-Base Mode")
    
    # --- 立脚開始・終了の検出 ---
    start_offset = search_window
    
    for i in range(start_offset, n_samples - search_window):
        
        # --- A. 立脚開始 (Heel Strike) ---
        if data[i] > threshold_Force and not in_stance:
            
            actual_start = i 
            
            # 1. 回帰に使うデータの「終点 (0.08V)」を探す
            #    i (0.1V) から過去に戻って探す
            idx_reg_end = i
            scan = i
            while scan > i - search_window:
                if data[scan] <= reg_voltage_end:
                    idx_reg_end = scan
                    break
                scan -= 1
            
            # 2. 回帰に使うデータの「始点 (0.03V)」を探す
            #    終点からさらに過去に戻って探す
            idx_reg_start = idx_reg_end
            scan = idx_reg_end
            while scan > i - search_window:
                if data[scan] <= reg_voltage_start:
                    idx_reg_start = scan
                    break
                scan -= 1
            
            # 3. 回帰計算 & 交点算出
            if (idx_reg_end - idx_reg_start) >= 2:
                x_reg = np.arange(idx_reg_start, idx_reg_end + 1)
                y_reg = data[idx_reg_start : idx_reg_end + 1]
                
                # 線形回帰 (y = slope * x + intercept)
                res = stats.linregress(x_reg, y_reg)
                slope = res.slope
                intercept = res.intercept
                
                # 0 = slope * x + intercept  =>  x = -intercept / slope
                if abs(slope) > 1e-6:
                    intersection_x = -intercept / slope
                    actual_start = int(np.round(intersection_x))
                    
                    # ガード処理: 
                    # 計算結果があまりに遠い過去(始点より200ms以上前)や、
                    # 未来(iより後)になった場合は、回帰の始点(idx_reg_start)を採用する
                    if not (idx_reg_start - 200 < actual_start < i):
                        actual_start = idx_reg_start
                else:
                    actual_start = idx_reg_start
            else:
                # データ点が足りない、または見つからない場合
                # とりあえず threshold_SearchLimit (0.03V) を切った点を採用
                fallback_scan = i
                while fallback_scan > i - search_window:
                    if data[fallback_scan] < threshold_SearchLimit:
                        actual_start = fallback_scan
                        break
                    fallback_scan -= 1

            stance_starts.append(actual_start)
            in_stance = True

        # --- B. 立脚終了 (Toe Off) ---
        elif data[i] < threshold_Force and in_stance:
            
            # 終了点はシンプルに threshold_SearchLimit を下回った点
            scan_end = i
            search_limit = min(n_samples, i + search_window)
            
            actual_end = i
            while scan_end < search_limit:
                if data[scan_end] < threshold_SearchLimit:
                    actual_end = scan_end
                    break
                scan_end += 1
            
            # さらにその先の最小値(完全に抜重した点)を探すとより丁寧
            local_search_end = min(n_samples, actual_end + 50) # 50ms先まで
            if actual_end < local_search_end:
                segment = data[actual_end : local_search_end]
                if len(segment) > 0:
                    min_idx = np.argmin(segment)
                    actual_end = actual_end + min_idx

            stance_ends.append(actual_end)
            in_stance = False

    # --- 3. ペアリング ---
    gait_cycles = []
    
    for start in stance_starts:
        valid_ends = [end for end in stance_ends 
                      if end > start and end - start < max_cycle_limit]
        
        if valid_ends:
            end = min(valid_ends)
            next_starts = [s for s in stance_starts if s > end]
            
            if next_starts:
                next_start = min(next_starts)
                
                gait_cycles.append({
                    'hs_time': start / 1000.0,
                    'to_time': end / 1000.0,
                    'next_hs_time': next_start / 1000.0,
                    'hs_frame': start,
                    'to_frame': end,
                    'next_hs_frame': next_start
                })
    
    print(f"\n[Zero-Base] 検出数: {len(gait_cycles)} 歩\n")
    return gait_cycles

def calculate_gait_cycles(data):
    if len(data) == 0:
        print("警告: 空のデータセットです。")
        return []
    
    # 微分値の計算
    diff_data = np.diff(data)
    
    # 閾値の設定
    rise_threshold = 0.0001   # 立ち上がりの閾値 0.0001
    fall_threshold = -0.0001  # 降下の閾値 -0.0001
    force_threshold = 0.09    # 力データの閾値 0.2
    window_size = 8         # 線形近似に使用するウィンドウサイズ 5
    max_cycle_duration = 1800  # 最大歩行周期時間（2秒 = 2000サンプル）
    
    stance_starts = []
    stance_ends = []
    
    # 立脚開始点の検出と補正
    for i in range(len(diff_data)-1):
        if (diff_data[i] > rise_threshold and 
            data[i] < force_threshold and 
            data[i+1] > force_threshold):
            # 検出点から前方のデータを使用して線形近似
            if i >= window_size:
                x = np.arange(i-window_size, i)
                y = data[i-window_size:i]
                slope, intercept, _, _, _ = stats.linregress(x, y)
                # x切片を計算（実際の立脚開始点）
                actual_start = int(-intercept/slope)
                if 0 <= actual_start < len(data):  # インデックスの範囲チェック
                    stance_starts.append(actual_start)
    
    # 立脚終了点の検出と補正
    for i in range(len(diff_data)-1):
        if (diff_data[i] < fall_threshold and 
            data[i] > force_threshold and 
            data[i+1] < force_threshold):
            # 検出点から後方のデータを使用して線形近似
            if i + window_size < len(data):
                x = np.arange(i-window_size, i)
                y = data[i-window_size:i]
                slope, intercept, _, _, _ = stats.linregress(x, y)
                # x切片を計算（実際の立脚終了点）
                actual_end = int(-intercept/slope)
                if 0 <= actual_end < len(data):  # インデックスの範囲チェック
                    stance_ends.append(actual_end)
    
   # 歩行周期のペアリング
    gait_cycles = []
    
    for start in stance_starts:
        # 現在の開始点以降で最も近い終了点を探す
        valid_ends = [end for end in stance_ends 
                     if end > start and 
                     end - start < max_cycle_duration]
        
        if valid_ends:
            end = min(valid_ends)  # 最も近い終了点を選択
           
            # 次の接地を探す
            next_starts = [s for s in stance_starts if s > end]
            if next_starts:
                next_start = min(next_starts)
                gait_cycles.append({
                    'hs_time': start / 1000.0,      # 接地時間 (秒)
                    'to_time': end / 1000.0,        # 離地時間 (秒)
                    'next_hs_time': next_start / 1000.0, # 次の接地時間 (秒)
                    'hs_frame': start,
                    'to_frame': end,
                    'next_hs_frame': next_start
                })
        
    # 検出結果の出力
    print(f"\n検出された歩行周期の総数: {len(gait_cycles)}\n")
    
    return gait_cycles

def calculate_gait_cycles_ver3(data):
    if len(data) == 0:
        print("警告: 空のデータセットです。")
        return []
    
    # ==========================================================================
    # ★設定パラメータ (定数)
    # ==========================================================================
    # 1. 存在判定用（確実に接地しているとみなす高い値）
    #    ノイズの最大値よりも十分に大きく設定してください。
    #    例: 20N〜50N相当, 正規化値なら 0.1(10%) など
    HIGH_THRESHOLD = 0.1
    fall_threshold = 0.0001

    # 2. タイミング特定用（0点付近の低い値）
    #    ここを基準に線を引きます。ノイズに埋もれないギリギリ低い値がベスト。
    LOW_THRESHOLD = 0.033

    # 3. 線形近似に使うウィンドウサイズ (フレーム数)
    #    Low閾値付近の「立ち上がり」の傾きをとるための幅
    REG_WINDOW = 6
    
    # 4. その他の制限
    MIN_CYCLE_DURATION = 500  # 最小歩行周期 
    MAX_CYCLE_DURATION = 1800  # 最大歩行周期 (サンプル数)
    BACK_SEARCH_LIMIT = 300    # High検知からLowを探して遡る最大フレーム数
    
    # ==========================================================================

    diff_data = np.diff(data)
    n_samples = len(data)
    
    stance_starts = []
    stance_ends = []
    
    # --- 1. 立脚開始点 (Heel Strike) の検出 ---
    # ロジック: High閾値を超えたら -> 過去に遡ってLow閾値を探す -> その付近で回帰
    
    for i in range(1, n_samples - 1):
        
        # High閾値をまたいだ瞬間（立ち上がり）を検知
        if (diff_data[i] >  fall_threshold and 
            data[i-1] < HIGH_THRESHOLD and 
            data[i] >= HIGH_THRESHOLD):
        
            high_idx = i
            found_low = False
            
            # そこから過去へ遡る (Low閾値を下回る点を探す)
            # max(0, ...) でインデックスのエラー防止
            search_limit = max(0, high_idx - BACK_SEARCH_LIMIT)
            
            for j in range(high_idx, search_limit, -1):
                if data[j] <= LOW_THRESHOLD:
                    # Low閾値付近に到達！ここを基準にする
                    low_idx = j
                    found_low = True
                    
                    # 線形近似 (Regression)
                    # Low閾値から「立ち上がる方向(未来)」のデータを使って直線を引く
                    # 区間: [low_idx, low_idx + REG_WINDOW]
                    if low_idx + REG_WINDOW < n_samples:
                        y_reg = data[low_idx : low_idx + REG_WINDOW]
                        x_reg = np.arange(low_idx, low_idx + REG_WINDOW)
                        
                        slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                        
                        # 傾きが正（右肩上がり）であることを確認
                        if slope > 1e-6:
                            # 0クロス点 (x = -b/a) を計算
                            actual_start = int(-intercept / slope)
                            
                            # 妥当性チェック (データ範囲内 かつ 極端に離れていない)
                            if 0 <= actual_start < n_samples:
                                # 重複防止 (念のため直近の検出と比較)
                                if not stance_starts or actual_start > stance_starts[-1] + 100:
                                    stance_starts.append(actual_start)

                    break # Lowが見つかったらループを抜ける
            
            if not found_low: pass

    # --- 2. 立脚終了点 (Toe Off) の検出 ---
    # ロジック: High閾値を下回ったら -> 未来へ進んでLow閾値を探す -> その付近で回帰
    
    for i in range(1, n_samples):
        # High閾値を下回った瞬間（降下）を検知
        if data[i-1] >= HIGH_THRESHOLD and data[i] < HIGH_THRESHOLD:
            
            high_idx = i
            found_low = False
            
            # そこから未来へ進む (Low閾値を下回る点を探す)
            search_limit = min(n_samples, high_idx + BACK_SEARCH_LIMIT)
            
            for j in range(high_idx, search_limit):
                if data[j] <= LOW_THRESHOLD:
                    low_idx = j
                    found_low = True
                    
                    # 線形近似 (Regression)
                    # Low閾値へ向かって「降りてくる方向(過去)」のデータを使って直線を引く
                    # 区間: [low_idx - REG_WINDOW, low_idx]
                    if low_idx - REG_WINDOW >= 0:
                        y_reg = data[low_idx - REG_WINDOW : low_idx]
                        x_reg = np.arange(low_idx - REG_WINDOW, low_idx)
                        
                        slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                        
                        # 傾きが負（右肩下がり）であることを確認
                        if slope < -1e-6:
                            actual_end = int(-intercept / slope)
                            
                            if 0 <= actual_end < n_samples:
                                if not stance_ends or actual_end > stance_ends[-1] + 100:
                                    stance_ends.append(actual_end)
                    break
    
    # --- 3. 歩行周期のペアリング ---
    gait_cycles = []
    
    for start in stance_starts:
        # startより後ろにある終了点を探す
        valid_ends = [end for end in stance_ends 
                      if end > start and end - start < MAX_CYCLE_DURATION]
        
        if valid_ends:
            end = min(valid_ends) # 最も近い終了点
            
            # startより後ろにある「次の開始点」を探す
            next_starts = [s for s in stance_starts if s > end]
            
            if next_starts:
                next_start = min(next_starts)
                
                gait_cycles.append({
                    'hs_time': start / 1000.0,
                    'to_time': end / 1000.0,
                    'next_hs_time': next_start / 1000.0,
                    'hs_frame': start,
                    'to_frame': end,
                    'next_hs_frame': next_start
                })
    
    print(f"\n検出された歩行周期の総数: {len(gait_cycles)}\n")
    return gait_cycles


def calculate_gait_cycles_ver5(fz_data, fy_data):
    if len(fz_data) == 0 or len(fy_data) == 0:
        return []
    
    # ==========================================================================
    # ★設定パラメータ
    # ==========================================================================
    # --- Fz用 (存在判定用) ---
    FZ_HIGH_THRESHOLD = 0.13  # 確実に体重が乗ったとみなす値
    FZ_LOW_THRESHOLD  = 0.033 # Fzの0点探索用
    
    # --- Fy用 (補正用) ---
    # Fyの "活動開始" を判定する閾値 (絶対値)
    # Fzで見つけた開始点において、Fyがこの値より大きければ「Fyの方が早い」とみなす
    FY_ACTIVE_THRESHOLD = 0.025
    
    # Fyで回帰直線を探すための探索閾値
    # ノイズに埋もれないギリギリの値 (Fzより少し敏感にしても良い)
    FY_LOW_THRESHOLD = 0.02

    # 共通設定
    REG_WINDOW = 6           # 線形近似の窓幅
    BACK_SEARCH_LIMIT = 300  # 遡る限界
    MIN_CYCLE_DURATION = 500
    MAX_CYCLE_DURATION = 1800
    # ==========================================================================

    diff_fz = np.diff(fz_data)
    n_samples = len(fz_data)
    
    stance_starts = []
    stance_ends = [] # 今回はStartの補正が主眼なのでEndはFz任せでOKとします
    
    # --- 1. 立脚開始点 (Heel Strike) の検出 ---
    
    for i in range(1, n_samples - 1):

        # A. Fzを使って「確実な一歩」を見つける (ここは変えない)
        if (diff_fz[i] > 0.0001 and 
            fz_data[i] < FZ_HIGH_THRESHOLD and 
            fz_data[i+1] >= FZ_HIGH_THRESHOLD):
        
            high_idx = i
            
            # ---------------------------------------------------------
            # B. まずは Fz でスタート地点を探す (基本ルート)
            # ---------------------------------------------------------
            fz_start_candidate = high_idx # 初期値
            found_fz_low = False
            
            search_limit = max(0, high_idx - BACK_SEARCH_LIMIT)
            
            # Fzを遡る
            for j in range(high_idx, search_limit, -1):
                if fz_data[j] <= FZ_LOW_THRESHOLD:
                    low_idx = j
                    found_fz_low = True
                    
                    # Fzで線形近似
                    if low_idx + REG_WINDOW < n_samples:
                        y_reg = fz_data[low_idx : low_idx + REG_WINDOW]
                        x_reg = np.arange(low_idx, low_idx + REG_WINDOW)
                        slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                        
                        if slope > 1e-6:
                            fz_start_candidate = int(-intercept / slope)
                    break
            
            # Fzで見つからなかったら、Highの位置を仮採用
            if not found_fz_low:
                print("not found!!!")
                fz_start_candidate = high_idx

            # ---------------------------------------------------------
            # C. Fyによる監査と補正 (Logic Update!)
            # ---------------------------------------------------------
            final_start = fz_start_candidate
            
            # Fzで見つけた開始点において、Fyがすでに動いているかチェック
            # インデックス範囲エラー防止
            check_idx = max(0, min(fz_start_candidate, n_samples - 1))
            
            if abs(fy_data[check_idx]) > FY_ACTIVE_THRESHOLD:
                
                fy_start_candidate = fz_start_candidate 
                
                # Fz開始点から過去へ遡る
                fy_search_limit = max(0, fz_start_candidate - 200)
                
                found_fy_low = False
                low_idx_fy = fz_start_candidate # 初期値
                
                # ★修正: argminではなく、ループで直近の「閾値割れ」を探す
                for j in range(fz_start_candidate, fy_search_limit, -1):
                    
                    # 絶対値が「静寂閾値」を下回ったら、そこが立ち上がり開始点とみなす
                    if abs(fy_data[j]) <= FY_LOW_THRESHOLD:
                        low_idx_fy = j
                        found_fy_low = True
                        
                        # 見つかったら即ループ終了！(これより過去は見ない)
                        break 
                
                # 見つかった場所を使って線形近似 (ここは今まで通り)
                if found_fy_low:
                    if low_idx_fy + REG_WINDOW < n_samples:
                        y_reg_fy = fy_data[low_idx_fy : low_idx_fy + REG_WINDOW]
                        x_reg_fy = np.arange(low_idx_fy, low_idx_fy + REG_WINDOW)
                        
                        slope_fy, intercept_fy, _, _, _ = stats.linregress(x_reg_fy, y_reg_fy)
                        
                        # 傾きがあればゼロクロス計算
                        if abs(slope_fy) > 1e-6:
                            calc_fy = int(-intercept_fy / slope_fy)
                            
                            # 計算結果が探索点の近くなら採用
                            if abs(calc_fy - low_idx_fy) < 50:
                                fy_start_candidate = calc_fy
                            else:
                                fy_start_candidate = low_idx_fy
                        else:
                            fy_start_candidate = low_idx_fy

                # 結果の採用判定
                if fy_start_candidate < fz_start_candidate:
                    if fz_start_candidate - fy_start_candidate < 200:
                        final_start = fy_start_candidate

            # ---------------------------------------------------------
            # D. 結果の格納
            # ---------------------------------------------------------
            if 0 <= final_start < n_samples:
                if not stance_starts or final_start > stance_starts[-1] + MIN_CYCLE_DURATION:
                    stance_starts.append(final_start)


    # --- 2. 立脚終了点 (Toe Off) は今回は省略 (既存ロジック推奨) ---
    for i in range(1, n_samples):
        # High閾値を下回った瞬間（降下）を検知
        if fz_data[i-1] >= FZ_HIGH_THRESHOLD and fz_data[i] < FZ_HIGH_THRESHOLD:
            
            high_idx = i
            found_low = False
            
            # そこから未来へ進む (Low閾値を下回る点を探す)
            search_limit = min(n_samples, high_idx + BACK_SEARCH_LIMIT)
            
            for j in range(high_idx, search_limit):
                if fz_data[j] <= FZ_LOW_THRESHOLD:
                    low_idx = j
                    found_low = True
                    
                    # 線形近似 (Regression)
                    # Low閾値へ向かって「降りてくる方向(過去)」のデータを使って直線を引く
                    # 区間: [low_idx - REG_WINDOW, low_idx]
                    if low_idx - REG_WINDOW >= 0:
                        y_reg = fz_data[low_idx - REG_WINDOW : low_idx]
                        x_reg = np.arange(low_idx - REG_WINDOW, low_idx)
                        
                        slope, intercept, _, _, _ = stats.linregress(x_reg, y_reg)
                        
                        # 傾きが負（右肩下がり）であることを確認
                        if slope < -1e-6:
                            actual_end = int(-intercept / slope)
                            
                            if 0 <= actual_end < n_samples:
                                if not stance_ends or actual_end > stance_ends[-1] + 100:
                                    stance_ends.append(actual_end)
                    break
    
    # --- 3. 歩行周期のペアリング ---
    gait_cycles = []
    
    for start in stance_starts:
        # startより後ろにある終了点を探す
        valid_ends = [end for end in stance_ends 
                      if end > start and end - start < MAX_CYCLE_DURATION]
        
        if valid_ends:
            end = min(valid_ends) # 最も近い終了点
            
            # startより後ろにある「次の開始点」を探す
            next_starts = [s for s in stance_starts if s > end]
            
            if next_starts:
                next_start = min(next_starts)
                
                gait_cycles.append({
                    'hs_time': start / 1000.0,
                    'to_time': end / 1000.0,
                    'next_hs_time': next_start / 1000.0,
                    'hs_frame': start,
                    'to_frame': end,
                    'next_hs_frame': next_start
                })
    
    print(f"\n検出された歩行周期の総数: {len(gait_cycles)}\n")
    return gait_cycles

def analyze_gait_phases(stance_starts, stance_ends, threshold=10):
    if not stance_starts or not stance_ends:
        print("警告: 歩行周期データが空です。")
        return []
    
    # サンプル数を秒に変換
    starts_sec = [s/1000 for s in stance_starts]
    ends_sec = [e/1000 for e in stance_ends]
    
    continuous_phases = []
    current_phase = [starts_sec[0]]
    current_ends = [ends_sec[0]]
    
    # 連続したフェーズの検出
    for i in range(1, len(starts_sec)):
        # 現在の開始点と直前の終了点との間隔を計算
        time_diff = starts_sec[i] - ends_sec[i-1]
        
        if time_diff <= threshold:
            # 連続とみなせる場合、現在のフェーズに追加
            current_phase.append(starts_sec[i])
            current_ends.append(ends_sec[i])
        else:
            # フェーズが途切れた場合の処理
            if len(current_phase) > 1:
                phase_start = min(current_phase)
                phase_end = max(current_ends)
                phase_duration = phase_end - phase_start
                
                # フェーズの妥当性チェック
                if (phase_duration >= threshold and  # フェーズ全体の長さが閾値以上
                    all(current_phase[j+1] - current_ends[j] <= threshold  # 全ての隣接する歩行周期間の間隔をチェック
                        for j in range(len(current_phase)-1))):
                    continuous_phases.append((phase_start, phase_end))
            
            # 新しいフェーズの開始
            current_phase = [starts_sec[i]]
            current_ends = [ends_sec[i]]
    
    # 最後のフェーズの処理
    if len(current_phase) > 1:
        phase_start = min(current_phase)
        phase_end = max(current_ends)
        phase_duration = phase_end - phase_start
        
        if phase_duration >= threshold:  # フェーズが10秒以上の場合のみ有効
            continuous_phases.append((phase_start, phase_end))
    
    # 検出結果の出力
    print("\n連続歩行フェーズ:")
    for i, (start, end) in enumerate(continuous_phases, 1):
        print(f"フェーズ {i}: {start:.2f}秒 - {end:.2f}秒 (継続時間: {end-start:.2f}秒)")
    return continuous_phases


def stand_estimate_BW(stand_path, r_fz_col, l_fz_col):
    """外部の静止立位ファイルから体重を推定"""
    print(f"\n静止立位データ '{os.path.basename(stand_path)}' から体重を推定します。")
    try:
        R_Fz = treadmill_data(stand_path, r_fz_col)
        L_Fz = treadmill_data(stand_path, l_fz_col)
        total_standing_force = np.mean(R_Fz) + np.mean(L_Fz)
        if total_standing_force <= 0:
            print("  警告: 合計荷重が0以下です。推定をスキップします。")
            return None
        return total_standing_force
    except Exception as e:
        print(f"  エラー: 外部静止立位ファイルの処理に失敗しました - {e}")
        return None
    
    
def save_average_csv(normalized_cycles, output_dir, base_filename, mode):
    """平均データをタスクフォルダ直下に保存"""
    if normalized_cycles is None: return
    
    if mode == 1: suffix = config.SUFFIX_STANCE_AVG
    else: suffix = config.SUFFIX_AVG

    save_filename = base_filename.replace('.csv', suffix)
    save_path = os.path.join(output_dir, save_filename)
   
    gait_percent = np.linspace(0, 100, 101)
    df_stats = pd.DataFrame({'Gait Cycle [%]': gait_percent})
    
    for force_key in ['Fx', 'Fy', 'Fz']:
        data_list = normalized_cycles.get(force_key, [])
        if not data_list: continue
        data_array = np.array(data_list)
        df_stats[f'{force_key}_Mean'] = np.mean(data_array, axis=0)
        df_stats[f'{force_key}_SD'] = np.std(data_array, axis=0)

    try:
        df_stats.to_csv(save_path, index=False, float_format='%.4f')
        print(f"★ 平均データ保存成功: '{save_path}'")
    except Exception as e:
        print(f"エラー: 平均データCSVの保存に失敗しました - {e}")

def filter_cycles_by_time(cycles_list, start_time, end_time, sampling_rate=1000):
    """周期リストから指定時間内のものだけを抽出する"""
    if start_time is None or end_time is None:
        return cycles_list
    
    s_frame = int(start_time * sampling_rate)
    e_frame = int(end_time * sampling_rate)
    
    filtered = []
    for cycle in cycles_list:
        if cycle['hs_frame'] >= s_frame and cycle['next_hs_frame'] <= e_frame:
            filtered.append(cycle)
    return filtered
