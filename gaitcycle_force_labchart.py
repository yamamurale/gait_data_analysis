#-----------------------------------------------
# gaitcycle_force_labchart.py　(ver3)
# LabChart解析用
# 右足・左足それぞれの床反力から歩行周期を算出し、個別のCSVに出力する
#
# 変更点
# 1．configの設定
# 2．ドリフト解消に用いる時間：config. ANALYSIS_START_TIME ~ 　（config.DRIFT_CORRECTION = true　時のみy方向もドリフト解消 ）
# 3. 体重推定にも値いる時間：config. STAND_START_TIME ~　config. STAND_END_TIME
# 4. 歩行周期csv出力名：タスク.txt名　＋　config.SUFFIX
# 5. グラフ解析データ　時間トリミング機能　ユーザー選択 　　(csv出力についてはトリミング機能非推奨（デフォルトFalse）)
# 6. 表示グラフ：立脚期のみ　or 全周期　ユーザー選択
#-----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data_processing 
import os
import config

def normalize_cycle(data, cycle_start, cycle_end, num_points=101):
    """歩行周期を指定されたサンプル数に正規化する"""
    cycle_data = data[cycle_start:cycle_end]
    x_new = np.linspace(0, 100, num_points)
    x_old = np.linspace(0, 100, len(cycle_data))
    normalized_data = np.interp(x_new, x_old, cycle_data)
    return normalized_data

def process_leg(filepath, leg_name, fx_col, fy_col, fz, output_filename, body_weight, analysis_start, analysis_end, use_trim_csv, use_trim_analysis, mode, output_dir):
    """
    片足分のデータを処理
    """
    print(f"\n========== {leg_name} Foot の処理を開始 ==========")
    
    # --- データの読み込み ---
    try:
        Fx = data_processing.treadmill_data(filepath, fx_col)
        Fy = data_processing.treadmill_data(filepath, fy_col)
        Fz = fz 
    except Exception as e:
        print(f"エラー: データの読み込みに失敗しました ({leg_name}) - {e}")
        return None, None

    # 静的バイアス補正
    if config.DRIFT_CORRECTION:
        bias_fy = data_processing.calc_static_bias_from_swing(Fy, Fz, analysis_start, duration=10.0, label=f"{leg_name}_Fy")
        Fy = Fy - bias_fy 

    # 体重推定
    if body_weight is None: body_weight = 1.0

    # 正規化
    Fx_norm = (Fx / body_weight) * 100
    Fy_norm = (Fy / body_weight) * 100
    Fz_norm = (Fz / body_weight) * 100

    # --- 歩行周期の検出 (全期間) ---
    print(f"  全データの歩行周期を検出中...")
   
    #all_gait_cycles = data_processing.calculate_gait_cycles_ver5(Fz,Fy) 
    all_gait_cycles = data_processing.calculate_gait_cycles_senior(Fz, force_threshold=0.09)

    if not all_gait_cycles:
        print(f"{leg_name}: 歩行周期が検出されませんでした。")
        return None, None
    

    # =========================================================
    # 1. 周期 保存用のデータ選定
    # =========================================================
    cycles_for_csv = []
    
    # use_trim_csv がTrueならトリミング、Falseなら全期間
    if use_trim_csv:
        print(f"  [Gait Cycle CSV] {analysis_start}-{analysis_end}秒 のデータのみ保存します。")
        cycles_for_csv = data_processing.filter_cycles_by_time(all_gait_cycles, analysis_start, analysis_end)
    else:
        print(f"  [Gait Cycle CSV] 全期間のデータを保存します。")
        cycles_for_csv = all_gait_cycles

    # CSV保存
    output_path = os.path.join(output_dir, output_filename)
    try:
        pd.DataFrame(cycles_for_csv).to_csv(output_path, index=False)
        print(f"★ 保存成功: '{output_path}'")
    except Exception as e:
        print(f"エラー: CSV保存失敗 - {e}")

    # =========================================================
    # 2. 解析 (平均・グラフ) 用のデータ選定
    # =========================================================
    cycles_for_analysis = []
    
    # use_trim_analysis がTrueならトリミング
    if use_trim_analysis:
        print(f"  [解析] {analysis_start}-{analysis_end}秒 のデータを使用して平均化・プロットします。")
        cycles_for_analysis = data_processing.filter_cycles_by_time(all_gait_cycles, analysis_start, analysis_end)
    else:
        print(f"  [解析] 全期間のデータを使用します。")
        cycles_for_analysis = all_gait_cycles

    # 歩行周期平均　＋　標準偏差
    if cycles_for_analysis:
        df_analysis = pd.DataFrame(cycles_for_analysis)
        # 時間差 (秒) を計算: next_hs_time - hs_time
        durations = df_analysis['next_hs_time'] - df_analysis['hs_time']
        
        mean_dur = durations.mean()
        std_dur = durations.std()
        
        print(f"  -> 解析対象: {len(cycles_for_analysis)} 歩")
        print(f"  ★ 歩行周期時間: {mean_dur:.4f} ± {std_dur:.4f} s")
    else:
        print(f"  -> 解析対象: 0 歩 (データなし)")

    # --- グラフ用正規化 ---
    normalized_cycles = {'Fx': [], 'Fy': [], 'Fz': []}
    if mode == 0: end_frame = 'next_hs_frame'
    elif mode == 1: end_frame = 'to_frame'

    for index, cycle in pd.DataFrame(cycles_for_analysis).iterrows():
        start, end = int(cycle['hs_frame']), int(cycle[end_frame])
        normalized_cycles['Fx'].append(normalize_cycle(Fx_norm, start, end))
        normalized_cycles['Fy'].append(normalize_cycle(Fy_norm, start, end))
        normalized_cycles['Fz'].append(normalize_cycle(Fz_norm, start, end))
    
    return normalized_cycles, cycles_for_analysis

def process_single_task(task_key, labchart_dir, stand_file, analyze_mode, save_avg = False):
    """1つのタスクを実行"""
    
    filename = config.TASKS[task_key]
    file_path = os.path.join(labchart_dir, filename)
    
    # タスク専用フォルダ
    task_output_dir = os.path.join(labchart_dir, task_key)
    os.makedirs(task_output_dir, exist_ok=True)
    
    print(f"\n--------------------------------------------------")
    print(f"  処理開始: {task_key} -> {task_output_dir}")
    print(f"--------------------------------------------------")

    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return None, None, None, None

    # --- 1. Raw CSV用の時間設定 (configで許可されている場合のみ聞く) 非推奨 ---
    use_trim_csv = False
    start_t = config.ANALYSIS_START_TIME
    end_t = config.ANALYSIS_END_TIME

    if config.ASK_TRIM_FOR_RAW_CSV:
        while True:
            try:
                print("\n------------------------------------------------")
                choice = int(input(f"[CSV保存設定] {start_t}秒～{end_t}秒 の期間のみを保存しますか？ (YES=1, NO=0 [全期間]): "))
                if choice == 1:
                    use_trim_csv = True
                    break
                elif choice == 0:
                    use_trim_csv = False
                    break
            except ValueError: pass
    else:
        # configがFalseなら全期間 (ユーザーには聞かない)
        use_trim_csv = False

    # --- 2. 解析(平均・グラフ)用の時間設定 (常に聞く) ---
    use_trim_analysis = False
    while True:
        try:
            print("\n------------------------------------------------")
            choice = int(input(f"[解析設定] グラフ表示に {start_t}秒～{end_t}秒 の期間のみを使用しますか？ (YES=1, NO=0 [全期間]): "))
            if choice == 1:
                use_trim_analysis = True
                break
            elif choice == 0:
                use_trim_analysis = False
                break
        except ValueError: pass

    # --- 体重推定 ---
    R_Fz_full = data_processing.adjusted_data(file_path, config.Rfz_col, config.ANALYSIS_START_TIME)
    L_Fz_full = data_processing.adjusted_data(file_path, config.Lfz_col, config.ANALYSIS_START_TIME)
    
    body_weight = None
    if stand_file:
        body_weight = data_processing.stand_estimate_BW(stand_file, config.Rfz_col, config.Lfz_col)
        print(f"[外部] 推定体重: {body_weight} V")
    else:
        s_t, e_t = config.STAND_START_TIME, config.STAND_END_TIME
        standing_force = np.mean(R_Fz_full[s_t*1000:e_t*1000]) + np.mean(L_Fz_full[s_t*1000:e_t*1000])
        if standing_force <= 0:
            print(f"警告: 立位荷重が0以下です。体重の計算を1.0で代用します。")
            body_weight = 1.0
        else:
            body_weight = standing_force
        print(f"[内部] 推定体重 ({s_t}-{e_t}s): {body_weight:.3f} V")
        
    # --- 出力ファイル名 ---
    file_name_right = f"{task_key}{config.SUFFIX_RIGHT}"
    file_name_left = f"{task_key}{config.SUFFIX_LEFT}"

    # --- 右足処理 ---
    # 2つのトリミングフラグを渡す
    right_cycles, _ = process_leg(
        file_path, "Right", config.Rfx_col, config.Rfy_col, R_Fz_full,
        file_name_right, body_weight, 
        start_t, end_t, use_trim_csv, use_trim_analysis, 
        analyze_mode, task_output_dir
    )

    # --- 左足処理 ---
    left_cycles, _ = process_leg(
        file_path, "Left", config.Lfx_col, config.Lfy_col, L_Fz_full,
        file_name_left, body_weight, 
        start_t, end_t, use_trim_csv, use_trim_analysis, 
        analyze_mode, task_output_dir
    )

    # --- 平均CSV保存 ---
    if save_avg and right_cycles and left_cycles:
        data_processing.save_average_csv(right_cycles, task_output_dir, file_name_right, analyze_mode)
        data_processing.save_average_csv(left_cycles, task_output_dir, file_name_left, analyze_mode)

    return right_cycles, left_cycles, file_path, task_output_dir

def main():
    print("--- LabChart 床反力正規化プログラム (左右対応版) ---")
    
    labchart_dir = os.path.join(config.BASE_DATA_DIR, config.LABCHART_DIR_NAME)
    stand_file = config.STAND_FILE_PATH

    while True:
        print("\n================================================")
        print("解析するタスクを選択してください:")
        
        available_tasks = []
        for i, task_key in enumerate(config.TASKS_TO_PROCESS):
            task_name = config.TASK_TITLES.get(task_key, task_key)
            print(f"  {i+1}: {task_name} ({task_key})")
            available_tasks.append(task_key)
        
        task_choice = input("番号を入力してください (q: 終了): ")
        if task_choice.lower() == 'q': 
            print("終了します。")
            break
        
        try:
            task_idx = int(task_choice) - 1
            if 0 <= task_idx < len(available_tasks):
                target_task = available_tasks[task_idx]
            else:
                print("無効な番号です。"); continue
        except ValueError: continue

        # --- 解析設定 ---
        while True:
            try:
                print("\n------------------------------------------------")
                analyze_mode = int(input(f"[{target_task}] グラフを立脚期のみに限定しますか？ (YES=1, NO=0): "))
                if analyze_mode in [0, 1]: break
            except ValueError: pass
        
        save_avg = False # 歩行周期平均をcsv出力したい場合　Trueに

        '''
        while True:
            try:
                print("\n------------------------------------------------")
                save_avg_choice = int(input(f"[{target_task}] 平均データ (Mean ± SD) のCSVも保存しますか？ (YES=1, NO=0): "))
                if save_avg_choice in [0, 1]: break
                else: print("1または0を入力してください。")         
            except ValueError: print("数値を入力してください。")
        '''

        # 実行
        r_cycles, l_cycles, f_path, output_dir = process_single_task(
            target_task, labchart_dir, stand_file, analyze_mode, save_avg
        )
        
        if r_cycles is None and l_cycles is None:
            print("解析に失敗しました。メニューに戻ります。")
            continue

        # グラフ表示ループ
        base_name = os.path.splitext(os.path.basename(f_path))[0]

        while True:
            # --- 1. ユーザーに表示する足を選択させる ---
            print("\n================================================")
            side_choice = input(f"[{target_task}] グラフ表示: 足を選択 (r: Right, l: Left, b: 戻る): ").lower()
            
            if side_choice == 'b': 
                break 
            
            plot_data = None
            leg_label = ""
            
            if side_choice == 'r':
                if r_cycles: plot_data, leg_label = r_cycles, "Right Foot"
                else:
                    print("エラー: 右足のデータがありません。")
                    continue # ループの先頭に戻る
            elif side_choice == 'l':
                if l_cycles: plot_data, leg_label = l_cycles, "Left Foot"
                else:
                    print("エラー: 左足のデータがありません。")
                    continue # ループの先頭に戻る
            else:
                print("無効な入力です。r, l, b のいずれかを入力してください。")
                continue # ループの先頭に戻る

            # スタイル選択
            while True:
                try:
                    style_input = input("プロットスタイルを選択 (1: グラデーション, 2: 平均±標準偏差): ")
                    style = int(style_input)
                    if style in [1, 2]: break
                    else: print("1か2を入力してください。")
                except ValueError: print("数値を入力してください。")

            # 描画先フォルダ: output_dir/plot/
            plot_dir = os.path.join(output_dir, 'plot')
            os.makedirs(plot_dir, exist_ok=True)
            
            style_name = "Gradient" if style == 1 else "Mean"
            graph_filename = f"{base_name}_{leg_label.replace(' ', '_')}_{style_name}.png"
            graph_path = os.path.join(plot_dir, graph_filename)

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f'Normalized GRF ({leg_label}) - {target_task}', fontsize=16)
            
            force_data = {
                'Fz (Vertical)': (axes[0], plot_data['Fz'], 'Blues'),
                'Fy (Anterior/Posterior)': (axes[1], plot_data['Fy'], 'Greens'),
            }
            x_axis = np.linspace(0, 100, 101)

            for label, (ax, data, cmap_name) in force_data.items():
                if not data: ax.text(0.5, 0.5, 'No Data'); continue
                
                disp_data = data
                # 波形グラフは反転なし
                
                if style == 1:
                    cmap = cm.get_cmap(cmap_name, len(disp_data))
                    for i, d in enumerate(disp_data):
                        ax.plot(x_axis, d, color=cmap(i/len(disp_data)), alpha=0.6)
                else:
                    data_arr = np.array(disp_data)
                    mean_d = np.mean(data_arr, axis=0)
                    std_d = np.std(data_arr, axis=0)
                    ax.plot(x_axis, mean_d, color='black', linewidth=2)
                    ax.fill_between(x_axis, mean_d-std_d, mean_d+std_d, alpha=0.3)
                
                ax.set_ylabel('Force (%BW)')
                ax.grid(True)
                ax.axhline(0, color='black', linewidth=0.5) 

            axes[-1].set_xlabel('Gait Cycle (%)')
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # --- 4. 画像保存の確認 ---
            while True:
                try:
                    print("\n------------------------------------------------")
                    save_choice = int(input(f"グラフを保存しますか？ (YES=1, NO=0): "))
                    if save_choice in [0, 1]: break
                    else: print("1または0を入力してください。")
                except ValueError: print("数値を入力してください。")

            if save_choice == 1:
                try:
                    plt.savefig(graph_path)
                    print(f"★ 保存成功: '{graph_path}'")
                except Exception as e:
                    print(f"エラー: グラフの保存に失敗しました - {e}")

            print("グラフを表示します。閉じる(×ボタン)と次の選択に戻ります。")
            plt.show()
            

if __name__ == '__main__':
    main()
