import datetime
import os
import sys
import time
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
import tkinter.ttk as ttk
from collections import OrderedDict
from distutils.command.config import config
from pathlib import Path
from re import I
from typing import List

import cv2
import japanize_matplotlib
import matplotlib as mpl
import numpy as np
import pandas as pd
from cv2 import line
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if __name__ == "__main__":
    import samos_utils as utl

else:
    from . import samos_utils as utl


class Element:
    """検出対象要素クラス"""

    def __init__(self, label: int, name: str = None, od_conf: float = 0, #type:ignore
                 xmin: int = 0, ymin: int = 0, xmax: int = 0, ymax: int = 0,
                 color: str = "#ffffff", classes: List[str] = list(), index: int = 0):
        """検出対象要素クラス

        Parameters
        ----------
        label : int
            物体検出時のラベル番号 \n
        name : str, optional
            要素名称, by default None\n
        conf : float, optional
            物体検出時の確信度, by default 0\n
        xmin : int, optional
            物体検出時のx座標の小さい方（左上）, by default 0\n
        ymin : int, optional
            物体検出時のy座標の小さい方（左上）, by default 0\n
        xmax : int, optional
            物体検出時のx座標の大きい方（右下）, by default 0\n
        ymax : int, optional
            物体検出時のy座標の大きい方（右下）, by default 0\n
        classes : List, optional
            各要素の画像分類の内容\n
            例：[cats,dogs,table], by default None\n
        ic_conf : np.ndarray, optional
            画像分類結果の確信度, by default None\n
        color : str, optional
            デバッグモードでの表示の際の表示色\n
            '#(xRR)(xGG)(xBB)' or 'colorcode'\n
            例："#ff007a","orange" by default "#ffffff"\n
        """

        self.label = label
        self.name = name
        self.od_conf = od_conf
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.color = color
        self.classes: List[str] = classes
        self.ic_conf = np.zeros(len(classes))
        self.frame_num: int = 0

    def show(self):
        print()
        print("label:", self.label)
        print("name:", self.name)
        print("conf:", self.od_conf)
        print("xmin:", self.xmin)
        print("ymin:", self.ymin)
        print("xmax:", self.xmax)
        print("ymax:", self.ymax)
        print("color:", self.color)
        print("class:", self.classes)
        print("ic_conf:", self.ic_conf)

class DataAnalysis(tk.Frame):
    """データ解析クラス

    work_inferenceで解析を実行し、出力された各種logデータを統合し、表示します。
    key_*.csvをベースに、時間とフレーム数を計算、動画ファイルとの整合性を取ります。
    デバッグモードでは、各要素においてどの作業が判定されているかも併せて表示します。

    Attribute:
        conf (samos_utils.Config): 作業情報設定ファイル
        master ([type], optional): ウィジェットを展開するためのtk.Tk(or tk.Frame). Defaults to None.
    """

    def __init__(self, master=None, debug_mode: bool = False):
        """データ解析クラス __init__

        Args:
            config (utl.Config): 作業情報設定ファイル
            master ([type], optional): ウィジェットを展開するためのtk.Tk(or tk.Frame). Defaults to None.
        """

        super().__init__(master)

        self.master = master #type:ignore
        self.debug_mode = debug_mode

        # * CONST
        # 画像描画
        self.VIS_WIDTH = 1280
        self.VIS_HEIGHT = 720

        self.UNIT_ROWS = 15

        # ステータス
        self.master_status_strings = [
            '停止', '準備中', '待機中',
            'OK', '注意', '警告',
            'システムエラー'
        ]
        self.master_color_codes = [
            'gray', 'orange', 'cyan',
            'limegreen', 'yellow', 'red',
            'purple'
        ]
        self.task_status_strings = [
            '未着手', '作業中', '作業中', '完了', 'エラー'
        ]
        self.task_color_codes = [
            'gray', 'yellow', 'yellowgreen', 'limegreen', 'red'
        ]
        self.ratio_color_codes = [
            'slate gray', 'dark goldenrod', 'green yellow', 'dark green', 'red'
        ]

        self.master_status_id = 0
        self.element_status_ids: List[int] = list()

        # * config情報
        self.config = utl.Config()
        self.elements_info: List[OrderedDict] = list()
        self.work_info: List[str] = list()
        self.process_info: List[OrderedDict] = list()
        # standard_times[work_id] (sec)
        self.standard_times: List[float] = list()

        # * time情報
        self.preparation_time = 0

        # * path
        self.rec_path = Path()
        self.prod_path = Path()

        # * movies
        self.capture: cv2.VideoCapture = cv2.VideoCapture()
        self.movie_fps: float = 30.0

        # * control
        self.cur_cycle = 0
        self.prev_cycle = 0
        self.playback_after_id = ""
        self.frame_ms = int(1/self.movie_fps*(10**3))
        self.task_durations = [0.00] * len(self.process_info)
        self.frame_count = 1000                

        # * 各種情報df
        self.integrated_log_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.stats_df = pd.DataFrame()
        self.time_df = pd.DataFrame()
        self.flow_df = pd.DataFrame()
        self.error_df = pd.DataFrame()

        # * 異常検出
        self.threesigma_cycles:List[int] = []
        self.error_cycles:List[int] = []

        # * TK
        # tk.frame
        self.header_frame = tk.Frame(self)
        self.movie_frame = tk.Frame(self)
        self.flow_frame = tk.Frame(self)
        self.data_frame = tk.Frame(self)
        self.line_frame = tk.Frame(self)
        # self.lower_frame = tk.Frame(self, relief=tk.GROOVE, bd=1)
        self.slider_frame = tk.Frame(self)
        self.control_frame = tk.Frame(self)
        self.pickup_frame = tk.Frame(self)
        # tk.variables
        self.svar_master_status = tk.StringVar(
            self, self.master_status_strings[0])
        self.svar_product_name = tk.StringVar(self, "")
        self.svar_worker_name = tk.StringVar(self, "未登録")
        self.svar_between = tk.StringVar(self, "開始時刻 ～ 終了時刻")
        self.ivar_total_cycle = tk.IntVar(self, 0)
        self.svar_time_at = tk.StringVar(self, "不明")
        self.ivar_cycle_at = tk.IntVar(self, 0)
        self.dvar_cycle_duration = tk.DoubleVar(self,0)
        self.svar_cycle_duration = tk.StringVar(self,"0.0")
        self.ivar_scale = tk.IntVar(self, 0)
        self.svar_scale = tk.StringVar(self, "0")
        self.svars_task_status: List[tk.StringVar] = list()
        self.svars_durations: List[tk.StringVar] = list()
        self.svars_time_ratio: List[tk.StringVar] = list()
        self.bln_y_invert = tk.BooleanVar(self,True)
        self.ivar_prepartion_time = tk.IntVar(self,0)

        # * flag
        self.config_is_loaded = False
        self.integrate_logs_is_loaded = False
        self.playback_flag = False

        # # * グラフまわり
        # mpl.rcParams['axes.xmargin'] = 0
        # mpl.rcParams['axes.ymargin'] = 0

        # * pack, widgetの作成
        self.pack()
        self.analysis()

    def create_widget(self):
        """
        widgetの作成

        tkinterに載せるウィジェットの作成・配置を行います。

        # TODO メニュー、header, progress_frame, lower_frameをそれぞれ作成する関数を作成すること（長くなり過ぎた）
        """

        # * メニューの作成
        menubar = tk.Menu(self)

        # フォント定義
        font_status = tkinter.font.Font(
            self, family="Meiryo UI", size=24, weight="bold")
        font_index = tkinter.font.Font(
            self, family="Meiryo UI", size=12, weight="bold")
        font_normal = tkinter.font.Font(
            self, family="Meiryo UI", size=10)
        font_bold = tkinter.font.Font(
            self, family="Meiryo UI", size=10, weight="bold")
        font_sub = tkinter.font.Font(
            self, family="Meiryo UI", size=8, slant="italic")

        # # ファイルメニュー
        # filemenu = tk.Menu(menubar, tearoff=0)
        # filemenu.add_command(label='解析の開始', command=self.analysis)
        # filemenu.add_separator()
        # filemenu.add_command(
        #     label='解析データを保存', command=self.save_integrated_log
        # )
        # filemenu.add_command(
        #     label="サマリーを保存", command=self.save_summary
        # )
        # filemenu.add_command(
        #     label="統計データを保存", command=self.save_stats
        # )
        # filemenu.add_command(
        #     label="チャートを保存", command=self.save_chart
        # )
        # filemenu.add_separator()
        # filemenu.add_command(label="終了", command=self.close)
        # menubar.add_cascade(label="ファイル", menu=filemenu)

        # # メニューバーの追加
        # self.master.configure(menu=menubar)  # type:ignore

        # * headerの定義
        # * 製品情報
        label_product_title = ttk.Label(
            self.header_frame,
            text="工程:",
            font=font_index
        )
        label_product_name = ttk.Label(
            self.header_frame,
            textvariable=self.svar_product_name,
            font=font_normal)  # 作業者情報
        label_worker_name_title = ttk.Label(
            self.header_frame,
            text="作業者:",
            font=font_index
        )
        label_worker_name = ttk.Label(
            self.header_frame,
            textvariable=self.svar_worker_name,
            font=font_normal
        )

        # * 時刻情報
        label_between_title = ttk.Label(
            self.header_frame,
            text="作業時間:",
            font=font_index
        )
        label_between = ttk.Label(
            self.header_frame,
            textvariable=self.svar_between,
            font=font_normal
        )
        # * サイクル情報
        label_total_cycle_title = ttk.Label(
            self.header_frame,
            text="総サイクル数：",
            font=font_index
        )
        label_total_cycle_num = ttk.Label(
            self.header_frame,
            textvariable=self.ivar_total_cycle,
            font=font_normal
        )

        # * 段取時間
        label_preparation_time_title = ttk.Label(
            self.header_frame,
            text="段取時間（秒）：",
            font=font_index
        )
        label_preparation_time_num = ttk.Label(
            self.header_frame,
            textvariable=self.ivar_prepartion_time,
            font=font_normal
        )

        # * grid
        label_product_title.grid(
            row=0, column=0,
            padx=5, sticky=tk.E
        )
        label_product_name.grid(
            row=0, column=1,
            padx=5, sticky=tk.E
        )
        label_worker_name_title.grid(
            row=0, column=2,
            padx=5, sticky=tk.E
        )
        label_worker_name.grid(
            row=0, column=3,
            padx=5, sticky=tk.W
        )
        label_between_title.grid(
            row=0, column=4,
            padx=5, sticky=tk.E
        )
        label_between.grid(
            row=0, column=5,
            padx=5, sticky=tk.E + tk.W
        )
        label_total_cycle_title.grid(
            row=0, column=6,
            padx=5, sticky=tk.E
        )
        label_total_cycle_num.grid(
            row=0, column=7,
            padx=5, sticky=tk.W
        )
        label_preparation_time_title.grid(
            row=0, column=8,
            padx=5, sticky=tk.E
        )
        label_preparation_time_num.grid(
            row=0, column=9,
            padx=5, sticky=tk.W
        )

        # * UpperFrame定義

        # * ビデオ描画領域の作成
        self.canvas_movie = tk.Canvas(
            self.movie_frame,
            width=self.VIS_WIDTH,
            height=self.VIS_HEIGHT,
            background="white"
        )
        self.canvas_movie.grid(
            row=0, column=0
        )

        # * master_statusの表示
        self.label_master_status = ttk.Label(
            self.data_frame,
            textvariable=self.svar_master_status,
            background=self.master_color_codes[self.master_status_id],
            font=font_status,
            anchor=tk.CENTER,
            relief=tk.GROOVE
        )
        self.label_master_status.grid(
            row=0, column=0,
            columnspan=3,
            sticky=tk.NSEW
        )

        # * ラベル
        label_work_title = ttk.Label(
            self.data_frame,
            text='作業',
            font=font_index)
        label_task_status_title = ttk.Label(
            self.data_frame,
            text='状態',
            font=font_index
        )

        label_work_title.grid(
            row=1, column=1,
            padx=5, sticky=tk.W
        )
        label_task_status_title.grid(
            row=1, column=2,
            padx=5, sticky=tk.W+tk.E
        )

        # * サイクル別データ
        label_cycle_at_title = ttk.Label(
            self.data_frame,
            text="サイクル:",
            font=font_index
        )
        label_cycle_at_val = ttk.Label(
            self.data_frame,
            textvariable=self.ivar_cycle_at,
            font=font_normal,
            width=3
        )
        label_duration_title = ttk.Label(
            self.data_frame,
            text="所要時間",
            font=font_index
        )

        label_cycle_at_title.grid(
            row=0, column=3,
            padx=10, sticky=tk.W
        )
        label_cycle_at_val.grid(
            row=0, column=4,
            sticky=tk.W
        )
        label_duration_title.grid(
            row=1, column=3, columnspan=2,
            padx=10, sticky=tk.W
        )

        # * 作業一覧の描画
        self.labels_status: List[ttk.Label] = list()
        self.labels_durations: List[ttk.Label] = list()

        for i, p in enumerate(self.process_info):

            # * 定義
            work_num_text = str(i+1)
            label_work_num = ttk.Label(
                self.data_frame,
                text=work_num_text,
                font=font_normal
            )

            work_name_text = p["work_name"]
            label_works = ttk.Label(
                self.data_frame,
                text=work_name_text,
                font=font_normal
            )
            
            self.labels_status.append(
                ttk.Label(
                    self.data_frame,
                    textvariable=self.svars_task_status[i],
                    background='#FFFFFF',
                    font=font_normal,
                    anchor='center',
                    width=6
                )
            )
            
            self.labels_durations.append(
                ttk.Label(
                    self.data_frame,
                    textvariable=self.svars_durations[i],
                    font=font_normal,
                    anchor='center',
                    width=6
                )
            )

            work_col = i // self.UNIT_ROWS
            work_row = i % self.UNIT_ROWS

            label_work_num.grid(
                row=3+work_row, column=work_col*4,
                padx=5, pady=3, sticky=tk.E
            )
            label_works.grid(
                row=3+work_row, column=work_col*4+1,
                padx=10, pady=3, sticky=tk.W
            )
            self.labels_status[i].grid(
                row=3+work_row, column=work_col*4+2,
                padx=10, pady=3, sticky=tk.W+tk.E
            )
            self.labels_durations[i].grid(
                row=3+work_row, column=work_col*4+3,
                padx=10, pady=3, sticky=tk.W
            )

        # サイクル
        label_total = ttk.Label(
            self.data_frame,
            text="サイクル",
            font=font_bold
        )
        label_total.grid(
            row=3+len(self.process_info), column=1,
            padx=10,pady=3,sticky=tk.E
        )

        label_cycle_duration = ttk.Label(
            self.data_frame,
            textvariable=self.svar_cycle_duration,
            font=font_normal,
            anchor='center',
            width=6
        )
        label_cycle_duration.grid(
            row=3+len(self.process_info), column=3,
            padx=10,pady=3,sticky=tk.W
        )

        # * 流動数グラフの作成
        self.init_flow_line()

        self.canvas_flow = FigureCanvasTkAgg(
            self.flow_fig,self.flow_frame)
        self.canvas_flow.draw()
        self.canvas_flow.get_tk_widget().grid(
            row=0, column=0)

        # * 各サイクル作業時間描画
        self.init_work_line()
        
        self.canvas_chart = FigureCanvasTkAgg(
            self.work_line_fig, self.line_frame)
        self.canvas_chart.draw()
        self.canvas_chart.get_tk_widget().grid(
            row=0, column=0,sticky=tk.NW
        )

        # * slider_frame
        scale = ttk.Scale(
            self.slider_frame,
            orient=tk.HORIZONTAL,
            variable=self.ivar_scale,
            from_=0,
            to=self.frame_count-1,
            length=1000,
            command=self.on_scale
        )
        lbl_time_val = ttk.Label(
            self.slider_frame,
            width=12,
            textvariable=self.svar_time_at,
            font=font_normal
        )
        lbl_scale_val = ttk.Label(
            self.slider_frame,
            width = 12,
            textvariable=self.svar_scale,
            font=font_sub
        )
        
        scale.grid(
            row=0, column=0,
            padx=5, pady=5, sticky=tk.W+tk.E
        )
        lbl_time_val.grid(
            row=0,column=1,
            padx=5,pady=5, sticky=tk.W
        )
        lbl_scale_val.grid(
            row=0, column=2,
            padx=5, pady=5, sticky=tk.W
        )

        # * control_frame
        self.btn_playback = tk.Button(
            self.control_frame,
            text="⏯", command=self.playback_control,
        )
        btn_large_back_alt = tk.Button(
            self.control_frame, 
            text="<<", 
            command=lambda:self.frame_control(-150))
        btn_small_back_alt = tk.Button(
            self.control_frame, 
            text="<",command=lambda:self.frame_control(-30))
        btn_small_step_alt = tk.Button(
            self.control_frame, 
            text=">",
            command=lambda:self.frame_control(30))
        btn_large_step_alt = tk.Button(
            self.control_frame, 
            text=">>", 
            command=lambda:self.frame_control(150))
        check_y_invert = tk.Checkbutton(
            self.control_frame,
            variable=self.bln_y_invert,
            command=self.graph_reload,
            text='Y軸反転'
        )
        
        self.btn_playback.grid(
            row=0, column=0,
            padx=5, pady=5, sticky=tk.W
        )
        btn_large_back_alt.grid(
            row=0, column=1, 
            padx=5, pady=5, sticky=tk.W
        )
        btn_small_back_alt.grid(
            row=0, column=2, 
            padx=5, pady=5, sticky=tk.W
        )                
        btn_small_step_alt.grid(
            row=0, column=3, 
            padx=5, pady=5, sticky=tk.W
        )                
        btn_large_step_alt.grid(
            row=0, column=4, 
            padx=5, pady=5, sticky=tk.W
        )
        check_y_invert.grid(
            row=0, column=5,
            padx=15,sticky=tk.W
        )

        # * pickup_frame

        pickup_note = ttk.Notebook(self.pickup_frame)

        self.threesigma_list = tk.Listbox(self.pickup_frame,height=20)
        for i,tsc in enumerate(self.threesigma_cycles):
            self.threesigma_list.insert(i,str(tsc))
        self.threesigma_list.bind(
            "<<ListboxSelect>>",
            self.jump_to_cycle_threesigma
        )

        self.error_list = tk.Listbox(self.pickup_frame)
        for i,ec in enumerate(self.error_cycles):
            self.error_list.insert(i,str(ec))
        self.error_list.bind(
            "<<ListboxSelect>>",
            self.jump_to_cycle_error
        )

        pickup_note.add(self.threesigma_list,text='3σ外サイクル')
        pickup_note.add(self.error_list,text='エラー発生サイクル')

        pickup_note.grid(row=0,column=0)
        
        # * 全体のgrid
        self.header_frame.grid(
            row=0, column=0, columnspan=2,
            padx=5, pady=5, sticky=tk.W
        )
        self.movie_frame.grid(
            row=1, column=0,
            padx=5, pady=5, sticky=tk.NSEW
        )
        self.data_frame.grid(
            row=1, column=1,
            padx=5, pady=5, sticky=tk.NSEW
        )
        self.pickup_frame.grid(
            row=1,column=2,
            padx=5,pady=5,sticky=tk.NSEW
        )
        self.flow_frame.grid(
            row=2, column=0,
            padx=5, pady=5, sticky=tk.NSEW
        )
        self.line_frame.grid(
            row=2,column=1, columnspan=2,
            padx=5,pady=5, sticky=tk.NSEW
        )
        self.slider_frame.grid(
            row=3, column=0, columnspan=3,
            padx=5, pady=5, sticky=tk.NSEW)
        self.control_frame.grid(
            row=4, column=0, columnspan=3,
            padx=5, pady=5, sticky=tk.W+tk.E
        )

    def analysis(self):
        """keyファイルの読込～サマリー情報の計算までを全て行います。"""
        self.load_key_file()
        self.movie_open(self.rec_path)
        # self.integrate_log()
        self.load_time_data()
        self.load_flow_data()
        self.load_status_data()
        self.load_error_data()
        # self.summarize_integrated_log()
        self.calc_stats()
        self.create_widget()
        self.update(0)

    def update(self, frame_num: int):
        """
        画面表示のアップデートを行います。

        各種tk.variablesにset等します。
        """

        update_start = time.perf_counter()

        # * 画像更新

        # movie_canvasのクリア
        self.canvas_movie.delete("fig")

        # 画像貼り付け
        img = self.load_img(frame_num)
        self.canvas_movie.photo = img  # type:ignore
        self.canvas_movie.create_image(
            0, 0, image=img,
            anchor='nw', tags='fig'
        )

        # * 指定フレームstatus読み込み
        self.ivar_scale.set(frame_num)
        self.svar_scale.set('フレーム:'+str(frame_num))
        cur_status_s = self.status_df.iloc[frame_num]
        
        # 時刻
        self.dt_time_at:datetime.datetime = cur_status_s['time'] #type:ignore
        str_time_at:str = self.dt_time_at.strftime('%H:%M:%S')
        self.svar_time_at.set(str_time_at)

        # サイクル
        self.cur_cycle:int = cur_status_s['cycle'] # type:ignore
        self.ivar_cycle_at.set(self.cur_cycle) # type:ignore

        # master_status
        cur_master_status_id = cur_status_s['master_status']
        cur_master_status_str = self.master_status_strings[cur_master_status_id]
        self.svar_master_status.set(cur_master_status_str)
        self.label_master_status.configure(
            background=self.master_color_codes[cur_master_status_id])
        
        # statuses
        cur_statuses = cur_status_s['statuses']
        
        # * グラフ更新

        # 流動線グラフ
        try:
            self.flow_vline.remove()
        except:
            pass
        self.flow_vline = self.flow_ax.axvline(self.dt_time_at,color='gray',linestyle='dashed') # type:ignore
        self.canvas_flow.draw()

        # サイクル作業時間グラフ
        try:
            self.work_vline.remove()
        except:
            pass
        self.work_vline = self.work_line_ax.axvline(self.dt_time_at,color='gray',linestyle='dashed') # type:ignore
        self.canvas_chart.draw()
                
        # * サイクル変化時
        if self.cur_cycle != self.prev_cycle:
            if self.cur_cycle > 0:

                if self.cur_cycle in self.time_df['unit'].values:
                    cur_cycle_durations = self.time_df[self.time_df.unit==self.cur_cycle]['duration']
                    for i,v in enumerate(cur_cycle_durations):
                        if i >0:
                            self.task_durations.append(v)
                    # self.task_durations = cur_cycle_durations.values.tolist() # type:ignore
                    self.draw_work_data(True)
                    self.canvas_chart.draw()

                else:
                    self.task_durations = [0.00] * len(self.process_info)
                
                try:
                    cycle_td:datetime.timedelta = self.cycle_durations[self.cur_cycle-1]
                    cycle_sec = cycle_td.total_seconds()
                    self.svar_cycle_duration.set(str(round(cycle_sec,2)))
                except:
                    self.svar_cycle_duration.set("0.00")

            else:
                self.task_durations = [0.00] * len(self.process_info)
                self.svar_cycle_duration.set("0.00")
                self.draw_work_data(False)
                self.canvas_chart.draw()

        # * 各タスク情報
        for i, t_s in enumerate(cur_statuses):
            
            # task_status
            task_status_str = self.task_status_strings[int(t_s)]
            self.svars_task_status[i].set(task_status_str)
            self.labels_status[i].configure(
                background=self.task_color_codes[int(t_s)]
            )

            # task_duration(actual)
            try:
                t_d = self.task_durations[i]
                self.svars_durations[i].set("{:.2f}".format(t_d))
            except:
                self.svars_durations[i].set("-")

        # * for playback
        update_end = time.perf_counter()
        td_update = int((update_end - update_start)*(10**3))
        skips = int(td_update/self.frame_ms)+1
        after_ms = self.frame_ms*skips - td_update

        if after_ms < 1:
            after_ms = 1

        self.prev_cycle = self.cur_cycle

        # 再生中は繰り返し
        if self.playback_flag:
            if frame_num < self.frame_count-skips:
                self.playback_after_id = self.after(
                    after_ms, self.update, frame_num+skips
                )
            else:
                self.playback_flag = False

    def load_img(self, frame_num:int):

        n = frame_num
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = self.capture.read()
        
        if ret:
            pil = utl.arr_to_pil(frame, is_bgr=True)
            img_itk = utl.pil_to_itk(pil, self)
            return img_itk
        else:
            return None

    def init_flow_line(self):
        
        # * データ整形
        output_x = [x for x in list(self.flow_df['output_x']) if pd.isnull(x)==False]
        output_y = [x for x in list(self.flow_df["output_y"]) if pd.isnull(x)==False]
        output_f = [x for x in list(self.flow_df['output_f']) if pd.isnull(x)==False]
        
        min_x = output_x[0]
        max_x = output_x[-1]
        min_f = output_f[0]
        max_f = output_f[-1]
        max_y = output_y[-1]

        self.ivar_total_cycle.set(int(max_y))
        
        input_x = [x for x in list(self.flow_df['input_x']) if pd.isnull(x)==False]
        input_y = [x for x in list(self.flow_df["input_y"]) if pd.isnull(x)==False]
        input_f = [x for x in list(self.flow_df['input_f']) if pd.isnull(x)==False]

        if min_x > input_x[0]:
            min_x = input_x[0]
        if max_x < input_x[-1]:
            max_x = input_x[-1]
        if min_f > input_f[0]:
            min_f = input_f[0]
        if max_f < input_f[-1]:
            max_f = input_f[-1]

        self.flow_min_x = mdates.date2num(min_x)
        self.flow_max_x = mdates.date2num(max_x)
        self.flow_x_range = self.flow_max_x - self.flow_min_x
        self.flow_min_f = min_f
        self.flow_max_f = max_f
        self.flow_f_range = self.flow_max_f - self.flow_min_f
            
        
        # * 描画
        self.flow_fig, self.flow_ax = plt.subplots()

        self.flow_ax.step(
            input_x,input_y,where='post',
            # marker="o",
            label="着手"
        )        
        
        self.flow_line, = self.flow_ax.step(
            output_x, output_y,where='post',
            # marker="o", 
            label="完了"
        )
        
        for i, row in enumerate(self.error_df.itertuples()):
            temp_time = row.time
            temp_error_type = row.error_type
            temp_color = 'black'
            if temp_error_type == 'program_error':
                temp_color = 'red'
            elif temp_error_type == 'irregular_work':
                temp_color = 'orange'
            elif temp_error_type == 'task_skip':
                temp_color = 'maroon'
            self.flow_ax.axvline(temp_time,color=temp_color,linestyle='dotted')


        # * 描画設定等
        self.set_size(5,2.8,self.flow_ax)

        self.flow_ax.set_xlabel("時刻")
        self.flow_ax.set_ylabel("台数")
        self.flow_ax.set_title("流動数グラフ")
        self.flow_ax.set_xlim(input_x[0],input_x[-1])
        self.flow_ax.set_ylim(0,input_y[-1]+1)

        if self.bln_y_invert.get():
            if not self.flow_ax.yaxis_inverted():
                self.flow_ax.invert_yaxis()

        self.flow_ax.legend()

        self.flow_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        # callback定義
        cid = self.flow_fig.canvas.mpl_connect( # type:ignore
            'button_press_event', self.flow_click
        )

    def flow_click(self,event):
        if event.inaxes != self.flow_line.axes:
            return
        else:
            if event.xdata > self.flow_max_x or event.xdata < self.flow_min_x:
                return

            if self.playback_flag:
                self.playback_flag=False
                self.after_cancel(self.playback_after_id)
            
            x_diff = event.xdata - self.flow_min_x
            ratio = x_diff/self.flow_x_range
            f_diff = int(self.flow_f_range * ratio)
            target_frame = int(self.flow_min_f + f_diff)
            
            self.update(target_frame)


    def init_work_line(self):
        """
        表示用チャートFigureを作成する

        以下の図を作成し、Figureインスタンスを返します。
        各サイクルの各作業の完了時間をプロットしたグラフを生成

        Returns:
            plt.Figure: matplotlib pyplotlibのFigureインスタンス
        """

        start = []
        finish = []
        y = []
        self.work_clickable = False
        self.work_min_x = datetime.datetime.now()
        self.work_max_x = datetime.datetime.now()
        self.work_min_f = 0
        self.work_max_f = 1

        self.work_line_fig, self.work_line_ax = plt.subplots()

        self.work_line_ax.cla()
        self.work_line_ax.plot(
            start,y,
            # marker='o',
            label='着手',color='gold')
        self.work_line, = self.work_line_ax.plot(
            finish,y,
            # marker='o',
            label="完了", color='lawngreen')

        self.set_size(5,2.8,self.work_line_ax)

        self.work_line_ax.set_xlabel("時刻")
        self.work_line_ax.set_ylabel("作業番号")
        self.work_line_ax.set_title("各サイクル 作業 開始/完了")
        self.work_line_ax.set_ylim(0,len(self.process_info)+1)

        # y軸反転
        if self.bln_y_invert.get():
            if not self.work_line_ax.yaxis_inverted():
                self.work_line_ax.invert_yaxis()

        self.work_line_ax.legend()
        
        cid = self.work_line_fig.canvas.mpl_connect( # type:ignore
            'button_press_event', self.work_click
        )

    def draw_work_data(self,draw_flag:bool):

        cycle_data = self.time_df[self.time_df['unit']==self.cur_cycle]
        error_data = self.error_df[self.error_df['cycle']==self.cur_cycle]

        if draw_flag:
            
            # * 階段グラフ
            start = cycle_data['start'].to_list()[0:]
            start_frame = cycle_data['start_frame'].to_list()[0:]
            finish = cycle_data['finish'].to_list()[0:]
            finish_frame = cycle_data['finish_frame'].to_list()[0:]
            y = cycle_data["work"].to_list()[0:]

            # 階段グラフ整形用
            start.append(finish[-1])
            start_frame.append(finish_frame[-1])
            finish.append(finish[-1])
            finish_frame.append(finish_frame[-1])
            y.append(y[-1])

            self.work_min_x = mdates.date2num(start[0])
            self.work_max_x = mdates.date2num(finish[-1])
            self.work_min_f = start_frame[0]
            self.work_max_f = finish_frame[-1]

            self.work_x_range = self.work_max_x - self.work_min_x
            self.work_f_range = self.work_max_f - self.work_min_f
         
            self.work_clickable = True

        else:
            start = []
            finish = []
            y = []
            self.work_clickable = False
            
            try:
                self.work_vline.remove()
            except:
                pass

        self.work_line_ax.cla()

        # * 階段グラフ
        self.work_line_ax.step(
            start,y,
            # marker='o',
            where='post',
            label='着手',color='gold')
        self.work_line, = self.work_line_ax.step(
            finish,y,
            # marker='o',
            where='post',
            label="完了", color='lawngreen')

        for e in error_data.itertuples():
            temp_error_type = e.error_type
            temp_color = 'black'
            if temp_error_type == 'program_error':
                temp_color = 'red'
            elif temp_error_type == 'irregular_work':
                temp_color = 'orange'
            elif temp_error_type == 'task_skip':
                temp_color = 'maroon'
            self.work_line_ax.axvline(e.time,color=temp_color,linestyle='dotted')
        
        self.work_vline = self.work_line_ax.axvline(self.dt_time_at,color='gray',linestyle='dashed') # type:ignore


        self.work_line_ax.set_xlabel("時刻")
        self.work_line_ax.set_ylabel("作業番号")
        self.work_line_ax.set_title('第' + str(self.cur_cycle) + "サイクル 作業")
        self.work_line_ax.set_ylim(0,len(self.process_info)+1)

        if self.bln_y_invert.get():
            if not self.work_line_ax.yaxis_inverted():
                self.work_line_ax.invert_yaxis()
        
        cid = self.work_line_fig.canvas.mpl_connect( # type:ignore
            'button_press_event', self.work_click
        )

        self.work_line_ax.legend()

    def work_click(self,event):
        if not self.work_clickable:
            return        
        if event.inaxes != self.work_line.axes:
            return
        else:
            if event.xdata > self.work_max_x or event.xdata < self.work_min_x:
                return

            if self.playback_flag:
                self.playback_flag=False
                self.after_cancel(self.playback_after_id)

            x_diff = event.xdata - self.work_min_x
            ratio = x_diff/self.work_x_range
            f_diff = int(self.work_f_range * ratio)
            target_frame = int(self.work_min_f + f_diff)
            
            self.update(target_frame)

    def graph_reload(self):

        if self.bln_y_invert.get():
            if not self.flow_ax.yaxis_inverted():
                self.flow_ax.invert_yaxis()
        else:
            if self.flow_ax.yaxis_inverted():
                self.flow_ax.invert_yaxis()

        if self.bln_y_invert.get():
            if not self.work_line_ax.yaxis_inverted():
                self.work_line_ax.invert_yaxis()
        else:
            if self.work_line_ax.yaxis_inverted():
                self.work_line_ax.invert_yaxis()

        self.canvas_flow.draw()
        self.canvas_chart.draw()

    def set_size(self, w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

    def jump_to_cycle_threesigma(self,event):
        index = self.threesigma_list.curselection()
        if len(index) == 1:
            cycle_num = int(self.threesigma_list.get(index))+1
            self.jump_to_cycle(cycle_num)

    def jump_to_cycle_error(self,event):
        index = self.error_list.curselection()
        if len(index) == 1:
            cycle_num = int(self.error_list.get(index))+1
            self.jump_to_cycle(cycle_num)

    def jump_to_cycle(self,cycle_num):
        input_f_s = self.flow_df['input_f']
        target_frame = int(input_f_s[cycle_num-1]) # type:ignore
        self.update(target_frame)

    def playback_control(self):

        if self.playback_flag:
            self.playback_flag = False
        else:
            self.playback_flag = True
            self.update(self.ivar_scale.get())

    def frame_control(self,step:int):

        if self.playback_flag:
            self.playback_flag = False
            self.after_cancel(self.playback_after_id)

        target_frame = self.ivar_scale.get() + step

        if step < 0:
            if target_frame < 0:
                target_frame = 0
        else:
            if target_frame >= self.capture.get(cv2.CAP_PROP_FRAME_COUNT):
                target_frame = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)-1)

        self.update(target_frame)

    def on_scale(self, val):
        """
        スライドバーが操作された際の関数

        得られた値から、

        Args:
            val ([type]): [description]
        """

        if self.playback_flag:
            self.playback_flag = False
            self.after_cancel(self.playback_after_id)

        frame_num = self.ivar_scale.get()
        self.svar_scale.set('フレーム:'+str(frame_num))
        
        self.update(frame_num)

    def load_config_file(self, path: Path = None): #type:ignore
        """
        設定データを読込ます。

        読込に成功した場合、必要なデータ（elem,work_name,work_order）を取得後、create_widgetをリロードします。
        """

        if path == None:
            config_path = self.config.load_data()
            if config_path == "":
                return None
        else:
            config_path = self.config.load_data(str(path))

        self.elements_info = self.config.data["element"]
        self.work_info = self.config.data["work_name"]
        self.process_info = self.config.data["work_order"]

        # 標準作業時間取得
        standard_info: dict = self.config.data["standard"]
        standard_df = pd.DataFrame()
        for elem, th in standard_info.items():
            s = pd.Series(th, name=elem)
            standard_df = pd.concat([standard_df, s], ignore_index=True)
        standard_df.sort_index(inplace=True,axis=1)
        
        standard_df = standard_df.sum()
        self.standard_times = standard_df.to_list()

        # process_infoの長さに応じたtk.variablesの定義
        for p in self.process_info:
            self.svars_task_status.append(
                tk.StringVar(self.data_frame, "未着手"))
            self.svars_durations.append(
                tk.StringVar(self.data_frame, "0.00"))
            self.svars_time_ratio.append(tk.StringVar(self.data_frame, "0.0"))

        self.config_is_loaded = True

    def load_key_file(self):
        """
        各種ログファイルの統合を行う

        movie_*.mp4とkey_*.csvをベースに、*の名前を持つ各ログファイル（elem,work,product）を統合します。
        統合されたデータは、dfとして保持します。
        dfのindexはmovieログのフレーム数で保持します。

        TODO 各種情報の読込のカプセル化（長くなり過ぎた）
        """

        tkinter.messagebox.showinfo("select key file",
                                    "解析対象のkey_fileを指定してください\nファイル名:key_*.csv")

        # keyファイル取得
        # # initial_dirの決定
        if not os.path.isdir("D:/SAMoS/log"):
            i_dir = os.path.abspath(os.path.dirname(__file__))
        elif "analysis_log_root_dir" not in self.config.data:
            i_dir = "D:/SAMoS/log/"
        elif self.config.data["analysis_log_root_dir"] == "":
            i_dir = "D:/SAMoS/log/"
        else:
            i_dir = os.path.abspath(os.path.dirname(
                self.config.data["analysis_log_root_dir"]+"/"))
        
        # # ファイルパス指定
        f_type = [("keyファイル", "*.csv")]
        key_path = tkinter.filedialog.askopenfilename(
            filetypes=f_type, initialdir=i_dir)

        # keyの読込
        try:
            key_info: dict = pd.read_csv(
                key_path, header=None, index_col=0).squeeze().to_dict()  # type:ignore
        except FileNotFoundError:
            print("File Not Found Error")
            tkinter.messagebox.showerror(
                "キーファイル読み込みエラー",
                "キーファイルが読み込めませんでした．\nプログラムを終了します")
            sys.exit()

        # # 作業者情報
        if "worker" in key_info: #type:ignore
            if key_info["worker"] != "":
                worker_name = key_info["worker"]
            else:
                worker_name = "未入力"
        else:
            worker_name = "不明"
        self.svar_worker_name.set(worker_name)

        # # 部品情報
        if "product" in key_info: # type:ignore
            if key_info["product"] != "":
                product_name = key_info["product"]
            else:
                product_name = "不明"
        else:
            product_name = "不明"
        self.svar_product_name.set(product_name)

        # # 開始・終了時間情報
        if "start" in key_info: # type:ignore
            if key_info["start"] != "":
                start_time = key_info["start"]
                start_str = datetime.datetime.strptime(
                    start_time, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S')
            else:
                start_str = "不明"
        else:
            start_str = "不明"

        if "end" in key_info: # type:ignore
            if key_info["end"] != "":
                end_time = key_info["end"]
                end_str = datetime.datetime.strptime(
                    end_time, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S')
            else:
                end_str = "不明"
        else:
            end_str = "不明"

        self.svar_between.set(start_str + " ～ " + end_str)

        # # config情報
        config_select_flag = True
        if self.config_is_loaded:
            config_select_flag = False
        else:
            if "config_path" in key_info: # type:ignore
                config_path = Path(key_info["config_path"])
                if config_path.exists():
                    self.load_config_file(config_path)
                    config_select_flag = False

        # # # config未設定時の場合に指定を要求
        if config_select_flag:
            self.load_config_file()

        # # movie情報
        movie_select_flag = True
        if "movie_path" in key_info: # type:ignore
            if key_info["movie_path"] != "":
                self.rec_path = Path(key_info["movie_path"])
                if self.rec_path.exists():
                    movie_select_flag = False

        # # # movie情報取得不可の場合
        if movie_select_flag:
            tkinter.messagebox.showinfo("select movie file",
                                        "解析対象の動画を指定してください\nファイル名:rec_*.mp4 (デフォルトの場合)")
            if "movie_rec_dir" not in self.config.data:
                i_dir = os.path.abspath(os.path.dirname(__file__))
            elif self.config.data["movie_rec_dir"] == "":
                i_dir = os.path.abspath(os.path.dirname(__file__))
            else:
                i_dir = os.path.abspath(os.path.dirname(
                    self.config.data["movie_rec_dir"]+"/"))
            f_type = [("動画ファイル", "*.mp4")]
            self.rec_path = Path(tkinter.filedialog.askopenfilename(
                filetypes=f_type, initialdir=i_dir))

        # # # prod情報
        # prod_select_flag = True
        # if "product_log_path" in key_info: # type:ignore
        #     if key_info["product_log_path"] != "":
        #         self.prod_path = Path(key_info["product_log_path"])
        #         if self.prod_path.exists():
        #             prod_select_flag = False

        # # # # prod情報取得不可の場合
        # if prod_select_flag:
        #     tkinter.messagebox.showinfo("select product log file",
        #                                 "解析対象の動画を指定してください\nファイル名:product_*.csv (デフォルトの場合)")
        #     if "product_log_dir" not in self.config.data:
        #         i_dir = os.path.abspath(os.path.dirname(__file__))
        #     elif self.config.data["product_log_dir"] == "":
        #         i_dir = os.path.abspath(os.path.dirname(__file__))
        #     else:
        #         i_dir = os.path.abspath(os.path.dirname(
        #             self.config.data["product_log_dir"]+"/"))
        #     f_type = [("ログファイル", "*.csv")]
        #     self.prod_path = Path(tkinter.filedialog.askopenfilename(
        #         filetypes=f_type, initialdir=i_dir))

        # * flow情報
        flow_select_flag = True
        if "flow_log_path" in key_info: # type:ignore
            if key_info["flow_log_path"] != "":
                self.flow_path = Path(key_info["flow_log_path"])
                if self.flow_path.exists():
                    flow_select_flag = False

        # # # flow情報取得不可の場合
        if flow_select_flag:
            tkinter.messagebox.showinfo("select flow log file",
                                        "解析対象の動画を指定してください\nファイル名:flow_*.csv (デフォルトの場合)")
            if "flow_log_dir" not in self.config.data:
                i_dir = os.path.abspath(os.path.dirname(__file__))
            elif self.config.data["flow_log_dir"] == "":
                i_dir = os.path.abspath(os.path.dirname(__file__))
            else:
                i_dir = os.path.abspath(os.path.dirname(
                    self.config.data["flow_log_dir"]+"/"))
            f_type = [("ログファイル", "*.csv")]
            self.flow_path = Path(tkinter.filedialog.askopenfilename(
                filetypes=f_type, initialdir=i_dir))

        # * time情報
        time_select_flag = True
        if "time_log_path" in key_info: # type:ignore
            if key_info["time_log_path"] != "":
                self.time_path = Path(key_info["time_log_path"])
                if self.time_path.exists():
                    time_select_flag = False

        # time情報取得不可の場合
        if time_select_flag:
            tkinter.messagebox.showinfo("select time log file",
                                        "解析対象の動画を指定してください\nファイル名:time_*.csv (デフォルトの場合)")
            if "time_log_dir" not in self.config.data:
                i_dir = os.path.abspath(os.path.dirname(__file__))
            elif self.config.data["time_log_dir"] == "":
                i_dir = os.path.abspath(os.path.dirname(__file__))
            else:
                i_dir = os.path.abspath(os.path.dirname(
                    self.config.data["time_log_dir"]+"/"))
            f_type = [("ログファイル", "*.csv")]
            self.time_path = Path(tkinter.filedialog.askopenfilename(
                filetypes=f_type, initialdir=i_dir))

        # * status情報
        status_select_flag = True
        if "status_log_path" in key_info: # type:ignore
            if key_info["status_log_path"] != "":
                self.status_path = Path(key_info["status_log_path"])
                if self.status_path.exists():
                    status_select_flag = False

        # # # status情報取得不可の場合
        if status_select_flag:
            tkinter.messagebox.showinfo("select status log file",
                                        "解析対象の動画を指定してください\nファイル名:status_*.csv (デフォルトの場合)")
            if "status_log_dir" not in self.config.data:
                i_dir = os.path.abspath(os.path.dirname(__file__))
            elif self.config.data["status_log_dir"] == "":
                i_dir = os.path.abspath(os.path.dirname(__file__))
            else:
                i_dir = os.path.abspath(os.path.dirname(
                    self.config.data["status_log_dir"]+"/"))
            f_type = [("ログファイル", "*.csv")]
            self.status_path = Path(tkinter.filedialog.askopenfilename(
                filetypes=f_type, initialdir=i_dir))

        # * error情報
        error_select_flag = True
        if "error_log_path" in key_info: # type:ignore
            if key_info["error_log_path"] != "":
                self.error_path = Path(key_info["error_log_path"])
                if self.error_path.exists():
                    error_select_flag = False

        # # # flow情報取得不可の場合
        if error_select_flag:
            tkinter.messagebox.showinfo("select error log file",
                                        "解析対象の動画を指定してください\nファイル名:error_*.csv (デフォルトの場合)")
            if "error_log_dir" not in self.config.data:
                i_dir = os.path.abspath(os.path.dirname(__file__))
            elif self.config.data["error_log_dir"] == "":
                i_dir = os.path.abspath(os.path.dirname(__file__))
            else:
                i_dir = os.path.abspath(os.path.dirname(
                    self.config.data["error_log_dir"]+"/"))
            f_type = [("ログファイル", "*.csv")]
            self.error_path = Path(tkinter.filedialog.askopenfilename(
                filetypes=f_type, initialdir=i_dir))


    def load_time_data(self):
        self.time_df = pd.read_csv(self.time_path, header=0,index_col=0)
        self.time_df['start'] = pd.to_datetime(self.time_df['start'])
        self.time_df['finish'] = pd.to_datetime(self.time_df['finish'])

    def load_flow_data(self):
        self.flow_df = pd.read_csv(self.flow_path)
        self.flow_df['output_x'] = pd.to_datetime(self.flow_df['output_x'])        
        self.flow_df['input_x'] = pd.to_datetime(self.flow_df['input_x'])

    def load_status_data(self):
        self.status_df = pd.read_csv(self.status_path,header=0,index_col=0)
        self.status_df['time'] = pd.to_datetime(self.status_df['time'])
        self.status_df['statuses'] = self.status_df['statuses'].str.strip('][').str.split(',')

    def load_error_data(self):
        self.error_df = pd.read_csv(self.error_path,header=0, index_col=0)
        self.error_df['time'] = pd.to_datetime(self.error_df['time'])
        
    def movie_open(self, movie_path: Path):
        self.capture = cv2.VideoCapture(str(movie_path))
        if not self.capture.isOpened():
            tkinter.messagebox.showerror(
                "読込失敗",
                "動画がの読み込みに失敗しました。\nプログラムを終了します。")
        # FPSの取得
        self.movie_fps = self.capture.get(cv2.CAP_PROP_FPS)
        # 総フレーム数の取得
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def calc_stats(self):
        """
        サマリーデータから各種統計データを計算します

        現在取得するもの：中央値・平均・標準偏差・最大・最小，段取時間
        """

        indexes = ['50%','mean','std','max','min']
        
        self.stats_df = pd.DataFrame(index=indexes)
        max_work = self.time_df['work'].max()
        
        # * 各作業
        for i in range(1,max_work+1):
            work_data = self.time_df[self.time_df['work']==i]
            durations = work_data['duration']
            describes = durations.describe()
            describes.rename(i,inplace=True)
            self.stats_df[i] = describes

        self.stats_df = self.stats_df.T

        # * サイクル
        max_unit = self.time_df['unit'].max()
        
        self.cycle_durations = []

        for n, (i,o) in enumerate(zip(self.flow_df['input_x'],self.flow_df['output_x'])):
            if n > 0:
                cycle_duration = o-i
                self.cycle_durations.append(cycle_duration)
            if n == 0:
                self.preparation_time = o-i
                preparation_sec = int(self.preparation_time.total_seconds())
                self.ivar_prepartion_time.set(preparation_sec)

        cycle_s = pd.Series(self.cycle_durations)
        c_describe = cycle_s.describe()

        self.cycles_describes = pd.Series()
        check_three_sigma_flag = True
        for idx in indexes:
            try:
                self.cycles_describes[idx] = c_describe[idx]
            except KeyError:
                print("標準偏差が計算できません。\n異常値チェックの機能は制限されます。")
                check_three_sigma_flag = False
                continue

        if check_three_sigma_flag:
            self.threesigma_cycles = self.pick_threesigma_cycle(self.cycle_durations, self.cycles_describes)
        self.error_cycles = self.pick_error_cycle(self.status_df)

    def pick_threesigma_cycle(self,cycle_durations:List[datetime.timedelta],cycle_describes:pd.Series):
        """
        3σ以上のサイクルを抽出します

        Args:
            cycle_durations (List[float]): サイクルの所要時間（timedelta）のリスト
            cycle_describes (pd.Series): サイクルの統計データのpandas.Series

        Returns:
            List[int]: 3σ以上のサイクルのサイクル番号を返します
        """        
        threesigma_cycles:List[int] = []
        mean = cycle_describes["mean"].total_seconds()
        std = cycle_describes['std'].total_seconds()

        for i, td in enumerate(cycle_durations):
            cd = td.total_seconds()
            diff = abs(cd - mean)
            sigma = diff/std
            if sigma > 3.0 :
                threesigma_cycles.append(i+1)

        # 抽出結果表示
        if len(threesigma_cycles) > 0:
            print('====3σ 外のサイクル====')
        for c in threesigma_cycles:
            print('第',c,'サイクル')
        
        return threesigma_cycles

    def pick_error_cycle(self,status_df:pd.DataFrame):
        """
        エラーの含まれるサイクルを抽出します

        Args:
            status_df (pd.DataFrame): フレームごとの状態を示すリスト

        Returns:
            List[int]: エラーを含むサイクルのサイクル番号を返します
        """        
        error_cycles:List[int] = []
        
        cycle_num = status_df['cycle'].max()

        for c in range(cycle_num+1):
            cycle_df = status_df.groupby('cycle').get_group(c)
            for status_s in cycle_df['statuses']:
                status_i = [int(s) for s in status_s]
                if 4 in status_i:
                    error_cycles.append(c)
                    break

        # 抽出結果表示
        if len(error_cycles) > 0:
            print('====エラーを含むサイクル====')
        for c in error_cycles:
            print('第',c,'サイクル')
        
        return error_cycles

    def save_integrated_log(self):
        """
        統合されたログを出力します

        出力ファイル名：analysislog_[title]_[YYMMDD_hhmmss].csv

        TODO key_log infoの書き込み
        TODO 新バージョン対応
        """

        filename = tkinter.filedialog.asksaveasfilename(
            title="名前を付けて保存",
            filetypes=[("csv", ".csv")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
            defaultextension="csv")

        try:
            self.integrated_log_df.to_csv(filename, encoding='utf-8_sig')
        except FileNotFoundError:
            print('File Not Found.')

    def save_summary(self):
        """
        サマリーデータを出力します

        output file:
            summary_[title]_[YYMMDD_hhmmss].csv
        output value:
            各サイクルの最大値
        TODO 新バージョン対応
        """

        filename = tkinter.filedialog.asksaveasfilename(
            title="名前を付けて保存",
            filetypes=[("csv", ".csv")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
            defaultextension="csv")

        try:
            self.summary_df.to_csv(filename, encoding='utf-8_sig')
        except FileNotFoundError:
            print('File Not Found.')

    def save_stats(self):
        """
        統計データを出力します

        output file:
            summary_[title]_[YYMMDD_hhmmss].csv
        output value:
            主となる作業者
            合計サイクル数
            合計作業数
            合計作業数中のNGの個数
            各作業のAve/SD/min/max
        TODO 新バージョン対応
        """

        filename = tkinter.filedialog.asksaveasfilename(
            title="名前を付けて保存",
            filetypes=[("csv", ".csv")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
            defaultextension="csv")

        try:
            self.stats_df.to_csv(filename, encoding='utf-8_sig')
        except FileNotFoundError:
            print('File Not Found.')

    def save_chart(self):

        # TODO 新バージョン対応

        filename = tkinter.filedialog.asksaveasfilename(
            title="名前を付けて保存",
            filetypes=[("png", ".png")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
            defaultextension="png")

        try:
            plt.savefig(filename,dpi=100,bbox_inches = 'tight', pad_inches = 0)
        except FileNotFoundError:
            print('File Not Found.')

    def close(self):
        """
        終了処理を行う

        TODO 保存されていない場合の警告
        """
        # ウィンドウを閉じる
        self.master.destroy()

        # mainの場合はtkinterごと落とす
        if __name__ == "__main__":
            self.master.quit()


if __name__ == "__main__":
    print('--- Data Analysis ---')

    # tkフレームの作成
    data_analysis_root = tk.Tk()
    # tkフレーム設定
    window_title: str = "Data Analysis | SAMoS DEBUG"
    data_analysis_root.title(window_title)
    data_analysis = DataAnalysis(master=data_analysis_root, debug_mode=True)  # type:ignore
    data_analysis_root.protocol('WM_DELETE_WINDOW', data_analysis.close)
    data_analysis_root.state("zoomed")


    # mainloopの開始
    data_analysis_root.mainloop()
