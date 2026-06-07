# =========================================================
# パスとファイル名の設定
# BASE_DATA_DIR\lab chart\
# └─ woexo_comf\  <-- タスク名のフォルダを自動作成
#     ├─ woexo_comf_gait_cycles_right.csv      (周期データ)
#     ├─ woexo_comf_gait_cycles_right_avg.csv  (平均データ) →　デフォルトなし
#     ├─ plot\                                 (床反力グラフ)
#     │   └─ ..._Mean.png
#     ├─ vector\                               (ベクトル図)
#     │   └─ vector_....png
#     └─ assist\                               (アシスト解析)
#         ├─ plot\
#         │   └─assist_....png
#         └─ woexo_comf_gait_cycles_left_assist.csv
# =========================================================

# データが格納されているルートディレクトリ
BASE_DATA_DIR = r"C:\data"

# LabChartのテキストデータ(.txt)があるフォルダ名
LABCHART_DIR_NAME = "lab chart"

# --- タスク定義 ---　（「タスク名」：「lab chartファイル名」）
TASKS = {
    'woexo_comf': 'woexo_comf.txt',
    'woexo_10ms': 'woexo_10ms.txt',
    'wexo_L_auto_comf': 'wexo_L_auto_comf.txt',
    'wexo_L_auto_10ms': 'wexo_L_auto_10ms.txt',
    'wexo_L_manual_comf': 'wexo_L_manual_comf.txt',
    'wexo_L_manual_10ms': 'wexo_L_manual_10ms.txt',
    'wexo_L_auto_10ms_2': 'wexo_L_auto_10ms_2.txt',
}

# --- タスクの表示名 ---（「タスク名」：「タイトル名」）
TASK_TITLES = {
    'woexo_comf': 'No Assist (PWS)',
    'woexo_10ms': 'No Assist (1.0 m/s)',
    'wexo_L_auto_comf': 'Auto (PWS)',
    'wexo_L_auto_10ms': 'Auto (1.0 m/s)',
    'wexo_L_manual_comf': 'Manual (PWS)',
    'wexo_L_manual_10ms': 'Manual (1.0 m/s)',
    'wexo_L_auto_10ms_2': 'Auto (1.0 m/s)',
}

# --- 解析対象タスク ---　
TASKS_TO_PROCESS = [
    'woexo_comf',
    'woexo_10ms',
    'wexo_L_auto_comf', 
    'wexo_L_auto_10ms',
    'wexo_L_manual_comf',
    'wexo_L_manual_10ms',
    'wexo_L_auto_10ms_2',
]

# --- 出力ファイル名のルール ---
SUFFIX_RIGHT = "_gait_cycles_right.csv"  # 歩行周期（右足）
SUFFIX_LEFT = "_gait_cycles_left.csv"    # 歩行周期（左足）
SUFFIX_AVG = "_avg.csv"                  # 周期平均の場合：SUFFIXファイルに_avg追加
SUFFIX_STANCE_AVG = "_stance_avg.csv"    # 周期平均 (立脚期のみ) の場合：SUFFIXファイルに_avg追加   

# =========================================================
# 2. 解析パラメータの設定
# =========================================================

# 外部立位タスクを用いて体重推定する場合は指定
STAND_FILE_PATH = "" 

STAND_START_TIME = 0       # 静止立位（体重推定） 開始[s]    
STAND_END_TIME = 10         #    〃　　　　　　　　終了[s]
ANALYSIS_START_TIME = 30   # 歩行解析 開始[s]  +  Fz ドリフト解消用計算（ANALYSIS_START_TIME～5秒間）
ANALYSIS_END_TIME = 60     # 歩行解析 終了[s] 

# memo 
# defalt : 150-290
# hori manual : 135-195
# hori manual : 135-195

DRIFT_CORRECTION = True   # y方向ドリフト解消の有無

# 【追加】歩行周期CSV(_gait_cycles_right.csv) を保存する際の設定
# True:  保存する時間を限定するかユーザーに聞く
# False: 常に全期間のデータをCSVに保存する (推奨)
ASK_TRIM_FOR_RAW_CSV = False

# =========================================================
# 3. LabChart チャンネル設定
# =========================================================
Rfx_col, Rfy_col, Rfz_col = 1, 2, 3
Lfx_col, Lfy_col, Lfz_col = 8, 9, 10

# =========================================================
# 4. グラフ・ベクトル図の描画設定
# =========================================================

# データ反転設定
FLIP_FX = False
FLIP_FY = True   #通常y方向データ反転
FLIP_FZ = False

# グラフ描画設定
VECTOR_FIGSIZE = (8, 8)
XLIM = (-50, 110)  #横軸   
FZ_YLIM = (-10, 170) #Fz
FY_YLIM = (-100, 80) #Fy
VECTOR_STEP = 2   

# 両足重ね合わせグラフの横軸範囲
XLIM_COMBINED_PCT = (-70, 170)  # 周期軸 [%]
XLIM_COMBINED_TIME = (-0.3, 1.7) # 時間軸 [s]
COMBINED_FIGSIZE = (11, 8)   # グラフサイズ

# 横軸・縦軸データ設定

MODE_TO_AXES = {
    0: ('Fy', 'Fz', 'Fy', 'Fz'),
    1: ('Fx', 'Fz', 'Fx', 'Fz'),
    2: ('Fx', 'Fy', 'Fx', 'Fy'),
}

# CV設定
YLIM_CV = (0, 25)

# =========================================================
# 5. アシスト解析設定 (assist_analysis.py用)
# =========================================================

# アシスト信号が入っているLabChartのチャンネル番号
ASSIST_CHANNEL = 15

# アシスト検知の閾値 (電圧 V)
ASSIST_THRESHOLD = 2.0 

# アシスト時間(ms)
ASSIST_DURATION_MS = 200
DELAY = 107

# 出力ファイル名の接尾辞
SUFFIX_ASSIST = "_assist.csv"

# 時系列グラフの表示範囲設定
WALK_START_TIME = 10       # 歩行データ 開始[s]
WALK_END_TIME = 60         #    〃　 終了[s]
ASSIST_PLOT_Y_RANGE = (-20, 100)   # 縦軸: タイミング (%)
# ヒストグラムの設定
ASSIST_HIST_BINS = 24        # ビンの数 (5%刻みなら20, 2%刻みなら50)
ASSIST_HIST_RANGE = (-20, 100) # 表示範囲 (%)

# アシストデータのフィルタリング閾値 [%]
# ベクトル表示　Mode 3 (Filtered) の際に、これより小さいタイミングのデータは除外
ASSIST_FILTER_MIN_PERCENT = 0  # デフォルトは0

# アシストタスク名
ASSIST_TASKS = {
    'wexo_L_auto_comf',
    'wexo_L_auto_10ms',
    'wexo_L_manual_comf',
    'wexo_L_manual_10ms',
    'wexo_L_auto_10ms_2',
}