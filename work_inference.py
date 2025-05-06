# -*- coding: utf-8 -*-
"""
[summary]

[extended_summary]

"""

import atexit
import colorsys
import copy
import datetime
import gc
import mmap
import os
import re
import sys
import threading
import time
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
import tkinter.ttk as ttk
import pymssql
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import List, Optional, OrderedDict
from unicodedata import name
from sqlalchemy import pool
from cv2 import exp

# tensorflowのInformationを非表示に
# tfのimport前に入れること。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import japanize_matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from natsort import natsorted
from PIL import Image, ImageGrab, ImageTk  # type:ignore
from pypylon import pylon
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model, model_from_json
from keras.optimizers import adam_v2
from keras.utils import img_to_array

if __name__ == "__main__" or __name__ == "work_inference":
    # import io_process # ! comment for dev
    import samos_utils as utl
    from SSD_pierluigiferrari.data_generator.object_detection_2d_data_generator import \
        DataGenerator
    from SSD_pierluigiferrari.data_generator.object_detection_2d_geometric_ops import \
        Resize
    from SSD_pierluigiferrari.data_generator.object_detection_2d_misc_utils import \
        apply_inverse_transforms
    from SSD_pierluigiferrari.data_generator.object_detection_2d_patch_sampling_ops import \
        RandomMaxCropFixedAR
    from SSD_pierluigiferrari.data_generator.object_detection_2d_photometric_ops import \
        ConvertTo3Channels
    from SSD_pierluigiferrari.keras_layers.keras_layer_AnchorBoxes import \
        AnchorBoxes
    from SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetections import \
        DecodeDetections
    from SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetectionsFast import \
        DecodeDetectionsFast
    from SSD_pierluigiferrari.keras_layers.keras_layer_L2Normalization import \
        L2Normalization
    from SSD_pierluigiferrari.keras_loss_function.keras_ssd_loss import SSDLoss
    from SSD_pierluigiferrari.models.keras_ssd300 import ssd_300
    from SSD_pierluigiferrari.ssd_encoder_decoder.ssd_output_decoder import (
        decode_detections, decode_detections_fast)

else:
    # from . import io_process # ! comment for dev
    from . import samos_utils as utl
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_data_generator import \
        DataGenerator
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_geometric_ops import \
        Resize
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_misc_utils import \
        apply_inverse_transforms
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_patch_sampling_ops import \
        RandomMaxCropFixedAR
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_photometric_ops import \
        ConvertTo3Channels
    from .SSD_pierluigiferrari.keras_layers.keras_layer_AnchorBoxes import \
        AnchorBoxes
    from .SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetections import \
        DecodeDetections
    from .SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetectionsFast import \
        DecodeDetectionsFast
    from .SSD_pierluigiferrari.keras_layers.keras_layer_L2Normalization import \
        L2Normalization
    from .SSD_pierluigiferrari.keras_loss_function.keras_ssd_loss import \
        SSDLoss
    from .SSD_pierluigiferrari.models.keras_ssd300 import ssd_300
    from .SSD_pierluigiferrari.ssd_encoder_decoder.ssd_output_decoder import (
        decode_detections, decode_detections_fast)

# io = io_process.io_process()# ! comment for dev


class Element:
    def __init__(self, label: int, name: str = None, od_conf: float = 0, #type:ignore
                 xmin: int = 0, ymin: int = 0, xmax: int = 0, ymax: int = 0,
                 color: str = "#ffffff", classes:List[str] = list(), index:int = 0):
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
        self.src_img_index = index
        self.frame_num:int = 0
        self.time = datetime.datetime.now()

    def show(self):
        print()
        print("label:",self.label)
        print("name:",self.name)
        print("conf:",self.od_conf)
        print("xmin:",self.xmin)
        print("ymin:",self.ymin)
        print("xmax:",self.xmax)
        print("ymax:",self.ymax)
        print("color:",self.color)
        print("class:",self.classes)
        print("ic_conf:",self.ic_conf)
        print("src_img_index:",self.src_img_index)
        print()

class ElementDrawer:
    
    def __init__(self, 
        element:Element, expand_ratio:float, 
        work_names:List[str], font:tkinter.font.Font):
        """
        推定された検出対象を画像上に描画するためのクラス
        
        Args:
            element (Element): 検出対象
            expand_ratio (float): 画像の拡大率
            work_names (List[str]): 検出対象で実行中の作業名
            font (tkinter.font.Font): 描画用フォント設定
        """
        
        self.x_min = element.xmin * expand_ratio
        self.y_min = element.ymin * expand_ratio
        self.x_max = element.xmax * expand_ratio
        self.y_max = element.ymax * expand_ratio
        self.color = element.color
        self.font = font
        
        name = element.name
        od_conf_text:str = name+'({:.3f})'.format(element.od_conf)
        max_index = element.ic_conf.argmax()
        
        class_str = element.classes[max_index]
        
        if class_str == "0":
            work_name = work_names[0]
            state_str = ""

        else:

            job = re.split('_',class_str)
            job[0] = int(job[0])
            work_name = work_names[job[0]]

            if job[1] == "a":
                state_str = "start"
            elif job[1] == "b":
                state_str = "finish"
            else:
                state_str = ""

        max_ic_conf = element.ic_conf[max_index]
        ic_conf_text:str = work_name + state_str + '({:.3f})'.format(max_ic_conf)
        self.drawn_text = od_conf_text + '\n' + ic_conf_text

    def draw(self, canvas:tk.Canvas):
        """
        画像上に推論された検出対象を描画する関数

        Args:
            canvas (tk.Canvas): 描画対象となるtkinterのCanvas
        """
        
        canvas.create_rectangle(
            self.x_min, self.y_min,
            self.x_max, self.y_max,
            outline=self.color,
            tags='elem'
        )
        canvas.create_text(
            self.x_min, self.y_min,
            text = self.drawn_text, 
            anchor = 'sw',
            font = self.font,
            fill = self.color,
            tags = 'elem'
        )


class WorkObserver:
    def __init__(self, element, work_order: List, threshold: dict, standards: dict, config: utl.Config, online: bool):
        """各作業の管理を行うクラス

        Parameters
        ----------
        task_names: List[str]
            作業名リスト
        work_order: List[dict]
            作業内容の辞書リスト
        threshold: dict[dict]
            各要素における各作業の検出秒数を収めた辞書の辞書
            Configより取得。
        """
        # ? master_status
        # ? 0: stop, start_flag = False
        # ? 1: in preparation, start_flag = True | is_ready = False
        # ? 2: waiting, flags = True | wait first task start
        # ? 3: OK, flags = True | No Problem
        # ? 4: caution, flags = True | any task is finished before previous task
        # ? 5: warning, flags = True | last task is finished without any task
        # ? 6: Error, flags = True | Something system trouble happen

        # ! モード切替フラグ管理
        
        #* 順序通りにのみ作業の判定を行うかどうか
        #* True：順序通りのみ作業判定
        #* False:作業順序問わず全てを同時に作業判定
        self.force_order_flag = False  
        
        #* 全ての作業が完了になった後に次の一台をチェックするかどうか
        #* True：全ての作業が完了しないと次の一台は開始されない
        #* False：最終作業が完了していれば次の一台の判定が開始される
        self.all_complete_check = False
        
        #* 分析/監視モード
        #* True:分析モード。


        # * CONSTANTS
        self.NEXT_WORK_THRESH = 1.5
        self.CONF_THRESH_RATIO = 0.1 # ある要素がある作業状態であると判定する確信度の下限％
        self.START_THRESH_RATIO = 0.1

        self.work_order = work_order
        self.num_tasks = len(work_order)

        self.worker = ""
        self.config = config

        self.online = online
        
        self.ret = True

        # * マスターの状態 [0: 停止, 1: 準備中, 2: 待機中, 3: OK, 4: 警告, 5: 注意]
        self.master_status: int = 0
        
        self.comp_flag = True
        self.work_count = 0

        # * 各作業の状態 [0: 未着手, 1: 開始, 2: 保留値到達, 3: 完了, -1: エラー]
        self.statuses: np.ndarray = np.zeros(
            self.num_tasks, dtype=np.int8)   # 各作業の状態

        # * 各作業においてエラーが発生した場合の反応[0: 反応なし, 1: 行灯点灯, 2: ライン停止]
        self.signal = 0

        # * 作業時間監視関連

        self.scores: np.ndarray = np.zeros(
            self.num_tasks*2, dtype=np.float32)     # 各作業のスコア

        self.process_start_time = datetime.datetime.now()
        self.process_duration: float = 0.    # 工程の作業開始からの時間
        self.task_start_time = [datetime.datetime.now()]*self.num_tasks
        self.task_finish_time = [datetime.datetime.now()]*self.num_tasks
        self.task_start_frame = [0]*self.num_tasks
        self.task_finish_frame = [0]*self.num_tasks
        self.task_duration: List[float] = [0.] * self.num_tasks  # 各作業の開始から完了までの時間
        self.times_cache = []
        self.task_is_started = [False] * self.num_tasks
        self.task_is_error = [False] * self.num_tasks

        self.proc_start_time:List[datetime.datetime] = []
        self.proc_start_nums:List[int] = []
        self.proc_start_frame:List[int] = []
        self.proc_comp_time:List[datetime.datetime] = []
        self.proc_comp_nums:List[int] = []
        self.proc_comp_frame:List[int] = []

        self.task_ids: List[str] = list()
        self.task_names: List[str] = list()
        self.responses: List[int] = list()
        for i in work_order:
            self.task_ids.append(str(i["work_id"]))
            self.task_names.append(i["work_name"])
            try:
                self.responses.append(i["response"])
            except KeyError:
                self.responses.append(1)
        
        self.responses_count = 0
        self.RESPONSE_TIMER = 30

        self.cur_edge:int = 0
    
        self.orders = []
        for e in element:
            for c in e["classes"]:
                if c != "0":
                    self.orders.append(c)
        self.orders.sort()
        
        # * 閾値設定
        df = pd.DataFrame()
        # read config threshold dict
        for elem, d in threshold.items():
            s = pd.Series(d, name=elem)
            df = pd.concat([df,s.to_frame().T])
        # read config standard dict
        for elem, std in standards.items():
            for key, val in std.items():
                # update threshold
                col_str = str(key)+'_a'
                df.at[elem,col_str] = val*self.START_THRESH_RATIO        
        # 各値の合計を計算
        self.comp_th = df.sum().sort_index().to_numpy()

        self.first_task_td = datetime.timedelta(microseconds=self.comp_th[0]*(10**6))
        self.last_task_td = datetime.timedelta(microseconds=self.comp_th[-1]*(10**6)) 

        # * シェアードメモリ関連
        try:
            self.mm = mmap.mmap(fileno=-1, length=20000, tagname="SAMoSSharedMemory", access=mmap.ACCESS_WRITE)
        except :
            try:
                self.mm = mmap.mmap(fileno=-1, length=20000, tagname="SAMoSSharedMemory", access=mmap.ACCESS_WRITE, create=True) #type:ignore
            except :
                print("シェアードメモリーの作成に失敗しました")
                tkinter.messagebox.showerror("シェアードメモリ","シェアードメモリの作成に失敗しました。\nプログラムを終了します。")
                self.ret = False

        # * sql関連
        self.DATA_SOURCE = 'localhost'
        self.INITIAL_CATALOG = 'MAIN_SQL'
        self.USER_ID = 'sqluser'
        self.PASSWORD = 'sqluser'
        self.APP_NAME = 'SAMoS Ver1.3_dev'
        if self.online:
            self.mypool = pool.QueuePool(self.getconn, max_overflow=10,pool_size=5)

    def update_scores(self, elements:List, time_delta:float):
        """Elementが持つic_confに基づいてscoreを更新する

        Parameters
        ----------
        elements : List[Element]
            ic_confを持つElementのリスト
        """

        conf_array:np.ndarray = np.zeros(self.num_tasks*2)

        for e in elements:
            for cls, conf in zip(e.classes,e.ic_conf):
                
                if cls == "0":
                    continue
                
                cls_name = re.split('_',cls)
                label = int(cls_name[0])-1
                if cls_name[1] == "a":
                    n = 0
                else:
                    n = 1
                order = (label)*2+n
                
                if n == 1:
                    status = self.statuses[label]
                    if status == 0 or status == 4:
                        conf = 0

                conf_array[order]=conf

        hotted_array:np.ndarray = np.where(conf_array > self.CONF_THRESH_RATIO,1,0) # type:ignore
        
        add_array = hotted_array * time_delta
        
        # * 解析制約条件あり版
        if self.force_order_flag:
            next_work_array = np.where(self.statuses==0)[0]
            if len(next_work_array)>0:
                next_work = next_work_array[0]
            else:
                next_work = 0

            cur_work_array = np.where(self.statuses==1)[0]
            if len(cur_work_array)>0:
                cur_work = cur_work_array[0]
            else:
                cur_work = 0

            # print('self.statuses:{}'.format(self.statuses))
            # print('next_work:{}, cur_work:{}'.format(next_work,cur_work))

            n_w_idx = next_work * 2
            c_w_idx = cur_work * 2 + 1

            self.scores[n_w_idx] = self.scores[n_w_idx] + add_array[n_w_idx]
            self.scores[c_w_idx] = self.scores[c_w_idx] + add_array[c_w_idx]

        # * 解析制約条件無し版
        else:
            self.scores = self.scores + add_array
            
        # for i in range(8):
        #     j = i + 1
        #     s = i * 2
        #     print(str(j)+":"+str(self.scores[s]))

    def update_states(self,frame_count):
        """
        各作業の状態を更新する関数

        Args:
            frame_count (_type_): フレーム数
        """

        # print(self.statuses)
        
        dt_now = datetime.datetime.now()

        for o, s, th in zip(self.orders, self.scores, self.comp_th): # type:ignore

            job = re.split("_",o)
            label = int(job[0]) - 1

            # * waiting(master_status:2) -> 作業開始判定(master_status:3)
            if self.master_status == 2:
                if self.force_order_flag:
                    if label == 0 and job[1] == "a":
                        if s > th:
                            self.start_reset(frame_count)
                else:
                    if label < self.num_tasks-1 and job[1] == "a":
                        if s > th:
                            self.start_reset(frame_count)

            # 状態の更新(解析中)
            if self.master_status >= 3:

                # if label > 4:
                #     print(s, th,self.statuses[label])

                if s > th:

                    if job[1] == "a" and (self.statuses[label] == 0 or self.statuses[label] == 4):

                        self.statuses[label] = 1
                        self.task_is_started[label] = True
                        idx = label*2
                        task_offset = datetime.timedelta(microseconds=self.comp_th[idx]*(10**6))
                        self.task_start_time[label] = dt_now-task_offset
                        self.task_start_frame[label] = frame_count

                        if label > 0:
                            self.prev_task_check(label,frame_count)

                    elif job[1] == "b" and self.statuses[label] == 1:

                        self.statuses[label] = 3
                        self.task_is_started[label] = False
                        self.task_is_error[label] = False
                        
                        idx = label*2+1
                        task_offset = datetime.timedelta(microseconds=self.comp_th[idx]*(10**6))
                        self.task_finish_time[label] = dt_now-task_offset
                        time_delta = self.task_finish_time[label] - self.task_start_time[label]
                        self.task_duration[label] = time_delta.total_seconds()
                        if self.task_duration[label]<0:
                            self.task_duration[label] = 0
                        self.task_finish_frame[label] = frame_count

                self.error_check(frame_count)
                
                # * debug self.signal 表示用
                if self.responses_count >= self.RESPONSE_TIMER:
                    # print("現在の出力信号:",self.signal, "\t[0:対応無し, 1:行灯点灯, 2:ライン停止]")
                    self.responses_count = 0
                else:
                    self.responses_count += 1

                # * 作業時間の計算
                for i, is_started in enumerate(self.task_is_started):
                    if is_started:
                        task_td = dt_now - self.task_start_time[i]
                        self.task_duration[i] = task_td.total_seconds()
        # サイクル完了判定
        self.completion_check(frame_count)

        # for n, t in zip (self.proc_comp_nums, self.proc_comp_time):
        #     print(n,t)

    def start_reset(self, frame):
        """
        新たな1台が開始された際に，作業一覧の状態をクリアする関数

        Args:
            frame (_type_): クリア時のフレーム番号
        """
        
        self.comp_flag = False
        self.master_status = 3
        self.work_count += 1
        self.scores[1:] = 0
        self.statuses[:] = 0

        start_dt = datetime.datetime.now()-self.first_task_td
        self.process_start_time = start_dt
        self.process_duration = 0.    # 工程の作業開始からの時間
        self.task_start_time:list = [start_dt]*self.num_tasks
        self.task_finish_time:list = [start_dt]*self.num_tasks
        self.task_start_frame = [frame]*self.num_tasks
        self.task_finish_frame = [frame]*self.num_tasks
        self.task_duration = [0.] * (self.num_tasks)
        self.task_is_started[:] = [False] * (self.num_tasks)

        self.proc_start_time.append(datetime.datetime.now())
        self.proc_start_nums.append(self.proc_start_nums[-1]+1)
        self.proc_start_frame.append(frame)

        if len(self.proc_comp_nums) == 0:
            self.proc_comp_time.append(datetime.datetime.now())
            self.proc_comp_nums.append(0)
            self.proc_comp_frame.append(frame)

        # if len(self.proc_comp_time) == 0:
        #     self.proc_comp_time.append(start_dt)
        #     self.proc_comp_nums.append(0)
        #     self.proc_comp_frame.append(frame)

        # print(self.work_count, "台目開始")

    def prev_task_check(self,label,frame_count):
            prev_label = label -1
            if self.statuses[prev_label] == 1:
                self.statuses[prev_label] = 3
                dt_now = datetime.datetime.now()
                delta_t = dt_now-self.task_start_time[prev_label]
                self.task_finish_time[prev_label] = dt_now
                self.task_finish_frame[prev_label] = frame_count
                self.task_is_started[prev_label] = False
                idx = prev_label*2+1
                self.task_duration[prev_label] = delta_t.total_seconds() - self.comp_th[idx]

    def error_check(self,frame_count):
        """
        もしひとつでもエラーがあれば工程内で開始されている最後の作業の位置を抽出
        
        Args:
            frame_count (_type_): フレーム数
        """

        # 現在のサイクルで完了となっている最後のタスクを検索
        cur_work_id = [i for i, x in enumerate(self.statuses) if x == 3] # type:ignore
        if cur_work_id != []:
            self.cur_edge = max(cur_work_id)
        else: 
            self.cur_edge = 0

        if self.cur_edge > 0:

            # 作業nが未開始，作業n以降が完了状態のときに作業nをエラー状態に変更
            self.statuses[0:self.cur_edge] = np.where(
                self.statuses[0:self.cur_edge] == 0, 
                4, self.statuses[0:self.cur_edge])

            # 現在のサイクルでエラーとなっているタスクを検索
            error_work_ids = [i for i, x in enumerate(self.statuses) if x==4] # type:ignore
            # エラーフラグを付与
            for e in error_work_ids:
                self.task_is_error[e] = True

            # 作業nが開始状態，作業n以降が完了状態の時に強制的に作業nを完了にする
            started_list = np.where(self.statuses[0:self.cur_edge] == 1)
            for i, s in enumerate(started_list):
                for c in s: #type:ignore
                    if self.task_is_error[c]:
                        continue
                    self.statuses[c] = 3
                    dt_now = datetime.datetime.now()
                    delta_t = dt_now-self.task_start_time[c]
                    self.task_finish_frame[c] = frame_count
                    self.task_is_started[c] = False
                    idx = c*2+1
                    self.task_duration[c] = delta_t.total_seconds() - self.comp_th[idx]
            # self.statuses[0:self.cur_edge] = np.where(self.statuses[0:self.cur_edge] == 1, 3, self.statuses[0:self.cur_edge])
            
            if self.cur_edge != self.num_tasks-1:
                self.statuses[self.cur_edge+1:] = np.where(
                    self.statuses[self.cur_edge+1:] == 4, 
                    0, self.statuses[self.cur_edge+1:])
            
        self.signal = 0

        if np.any(self.statuses == 4):
            if self.statuses[-1] == 3:  # 完了作業が工程の最終作業の場合
                self.master_status = 5  # * warning
            else:
                self.master_status = 4  # * caution

            # * エラー発生時の反応を発出
            response_mask:list[bool] = (self.statuses == 4) # type:ignore
            for i, k in enumerate(response_mask):
                if k:
                    if self.signal < self.responses[i]:
                        self.signal = self.responses[i]

        else:
            self.master_status = 3

        self.completion_check(frame_count)

    def completion_check(self,frame_count):
        """
        全作業が完了しているかのチェックを行い，Trueならばサイクルの完了を行う

        Args:
            frame_count (_type_): フレーム数
        """

        # サイクル完了判定
        if not self.comp_flag:
            # 工程の作業時間の計算
            end_dt = datetime.datetime.now()-self.last_task_td
            if self.work_count > 0:
                process_td = end_dt-self.process_start_time
                self.process_duration = process_td.total_seconds()

            # 全作業完了時にコンプリートとする場合
            if self.all_complete_check:
                # 全作業が保留（２）以上かつエラーが無ければ全て完了とし、待機状態に移行
                if np.all(self.statuses == 3) and not np.any(self.statuses == 4):  # type:ignore
                    self.comp_flag = True
                    self.master_status = 2
                    self.scores[:] = 0
                    self.statuses[:] = 3
                    self.proc_comp_time.append(end_dt)
                    self.proc_comp_nums.append(self.work_count)
                    self.proc_comp_frame.append(frame_count)
                    self.times_cache.append([self.task_start_time,
                        self.task_finish_time,self.task_duration,
                        self.task_start_frame,self.task_finish_frame])
                    self.cycle_report()
                    
            # 最終作業完了時にコンプリートとする場合
            else:
                if self.statuses[-1] == 3:
                    self.comp_flag = True
                    self.master_status = 2
                    self.scores[:] = 0
                    self.statuses[:] = 3
                    self.proc_comp_time.append(end_dt)
                    self.proc_comp_nums.append(self.work_count)
                    self.proc_comp_frame.append(frame_count)
                    self.times_cache.append([self.task_start_time,
                        self.task_finish_time,self.task_duration,
                        self.task_start_frame,self.task_finish_frame])
                    self.cycle_report()
                
    def force_complete(self, id:int, frame_count):
        """
        該当作業を強制的に完了にする関数

        Args:
            id (int): 該当の作業番号
            frame_count (_type_): フレーム数
        """
        
        if self.comp_flag:
            self.start_reset(frame_count)
        self.statuses[id] = 3
        self.task_finish_time[id] = datetime.datetime.now()
        self.task_finish_frame[id] = frame_count
        # start系がデフォルト値だった場合
        if id > 0 and self.task_start_time[id] == self.task_start_time[0]:
            self.task_start_time[id] = datetime.datetime.now()
            self.task_start_frame[id] = frame_count
        self.error_check(frame_count)
        self.completion_check(frame_count)

    def force_cancel(self,label:int,frame_count):
        """
        該当作業を強制的に取り消す関数

        Args:
            label(int): 該当の作業番号
            frame_count (_type_): フレーム数
        """
        
        self.statuses[label] = 0
        self.scores[label*2] = 0
        self.scores[label*2+1] = 0
        self.error_check(frame_count)

    def cycle_report(self):
        """
        サイクル完了時に，そのサイクルの統計データを出力
        """      

        print("=================")
        
        print("第",self.work_count,"サイクル")
        print("開始:",self.proc_start_time[-1],
              "完了:", self.proc_comp_time[-1],
              "所要時間（秒）:", self.process_duration)
        print("-----------------")
        
        for i, (wo, tst, tft, td) in enumerate(zip(self.work_order,
                                    self.task_start_time, 
                                    self.task_finish_time,
                                    self.task_duration)):
            print("作業",i+1,":", wo["work_name"])
            print("開始:",tst,"完了:",tft,"所要時間（秒）:",td)
        print("-----------------")
        
        print("作業者:",self.worker)

        abs_config_path = str(Path(self.config.file_path).absolute())

        self.mm_write(config_path = abs_config_path,
                      durations = self.task_duration)

        if self.online:
            self.sql_write(recipe=self.config.data['title'],
                        durations=self.task_duration,
                        worker_ID='')

    # * シェアードメモリ関連関数

    def cleanup(self):
        if not self.mm.closed:
            self.mm.close()

    def mm_write(self, config_path:str, durations:List[float]):

        shift_jis_path = str.encode(config_path,encoding='shift_jis')
        for i, c in enumerate(shift_jis_path):
            pos = 4 + i
            self.mm[pos] = c

        for x in range(len(durations)):
            pos = x * 4 + 1000
            int_duration_ms = int(durations[x]*1000)
            if int_duration_ms < 0:
                int_duration_ms = 0
            self.mm[pos : pos + 4] = int_duration_ms.to_bytes(4, 'little')

        #データ格納完了フラグ
        self.mm[0 : 4] = (1).to_bytes(4, 'little')

    # * SQL関連関数
    def getconn(self):
        con = pymssql.connect("localhost", "sqluser", "sqluser", "MAIN_SQL", appname = "SAMoS Ver1.3_dev", autocommit = True) #type:ignore
        return con

    def dataget(self, mypool, query):
        con = mypool.connect()
        cursor = con.cursor()  
        cursor.execute(query)
        data = tuple(cursor)
        con.close()
        return data

    def sql_write(self,recipe:str,durations:List[float],worker_ID:str =''):

        query  = "INSERT INTO     作業時間履歴概要\n"
        query += "               (日時, レシピ名, 社員NO)\n"
        query += "  OUTPUT        inserted.NO\n"
        query += "  VALUES       ('" + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') + "',\n"
        query += "                '" + recipe + "',\n"
        query += "                '" + worker_ID + "')"

        print(query)

        rows = self.dataget(self.mypool, query)

        getno = rows[0][0]

        query  = "INSERT INTO     作業時間履歴詳細\n"
        query += "               (NO, 工程NO, 作業時間)\n"
        query += "  VALUES\n"
        for i ,d in enumerate(durations):
            str_duration_ms = str(int(d*1000))
            if i < len(durations)-1:
                query += "               (" + str(getno) + ", " + str(i+1) + ", " + str_duration_ms + "),\n"
            else:
                query += "               (" + str(getno) + ", " + str(i+1) + ", " + str_duration_ms + ")"

        print(query)

        self.dataget(self.mypool, query)

class ElementDetector:
    """要素抽出担当クラス"""

    def __init__(self, config: utl.Config, src_width: int = None, src_height: int = None, threshold: float = 0.4): #type:ignore
        """[summary]

        Parameters
        ----------
        config : Config
            コンフィグ情報。samos_utilsで定義。
            configファイルは.json。（default_name: config.json）
        src_width : int, optional
            入力元画像幅。
            値が入力されなかった場合には、configのデータを利用する, by default None
        src_height : int, optional
            入力元画像高さ。
            値が入力されなかった場合には、configのデータを利用する, by default None
        conf_threshold : float, optional
            検出結果を確信度でフィルタする際の閾値, by default 0.6
        use_compressed_sample : bool, optional
            サブサンプリングした重みを利用しているか否か, by default False
        """

        self.threshold = threshold
        self.config = config

        # configデータ読み取り
        self.n_classes = config.data["elem_num"]
        self.e_names, self.e_widths, self.e_heights, self.e_nums = config.get_element_info()
        if src_width != None:
            self.width = src_width
        else:
            self.width = config.data['src']['width']
        if src_height != None:
            self.height = src_height
        else:
            self.height = config.data["src"]['height']

        # sort 準備
        self.columns = ['label', 'conf', 'xmin', 'ymin', 'xmax', 'ymax']

        # 要素ごとの描画色定義
        self.color_codes: List[str] = list()
        hues = np.linspace(0, 1, self.n_classes, endpoint=False)
        for i in range(self.n_classes):
            rgb_float = colorsys.hsv_to_rgb(hues[i], 1., 1.)
            rf, gf, bf = rgb_float
            r = int(rf * 255)
            g = int(gf * 255)
            b = int(bf * 255)
            rgb_code = '#' + format(r, '02x') + \
                format(g, '02x') + format(b, '02x')
            self.color_codes.append(rgb_code)

        # Set the image size.
        self.img_height: int = 300
        self.img_width: int = 300
        self.img_channels: int = 3

        # 1: Build the Keras model
        self.model: tf.keras.models.Model = ssd_300(image_size=(self.img_height, self.img_width, self.img_channels),  # type:ignore
                                                    n_classes=self.n_classes,
                                                    mode='inference',
                                                    l2_regularization=0.0005,
                                                    # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                                                    scales=[0.1, 0.2, 0.37,
                                                            0.54, 0.71, 0.88, 1.05],
                                                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                                             [1.0, 2.0, 0.5,
                                                                              3.0, 1.0/3.0],
                                                                             [1.0, 2.0, 0.5,
                                                                              3.0, 1.0/3.0],
                                                                             [1.0, 2.0, 0.5,
                                                                              3.0, 1.0/3.0],
                                                                             [1.0, 2.0,
                                                                              0.5],
                                                                             [1.0, 2.0, 0.5]],
                                                    two_boxes_for_ar1=True,
                                                    steps=[8, 16, 32,
                                                           64, 100, 300],
                                                    offsets=[0.5, 0.5, 0.5,
                                                             0.5, 0.5, 0.5],
                                                    clip_boxes=False,
                                                    variances=[
                                                        0.1, 0.1, 0.2, 0.2],
                                                    normalize_coords=True,
                                                    subtract_mean=[
                                                        123, 117, 104],
                                                    swap_channels=[2, 1, 0],
                                                    confidence_thresh=0.5,
                                                    iou_threshold=0.45,
                                                    top_k=200,
                                                    nms_max_output_size=400)

        # 2: Load the trained weights into the model.
        weights_path: str = config.data["elem_weight"]
        self.model.load_weights(weights_path, by_name=True)  # type:ignore

        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = adam_v2.Adam()
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.model.compile( # type:ignore
            optimizer=adam, loss=ssd_loss.compute_loss) #type:ignore

    def detection(self, arrays: np.ndarray):
        """SSDによる物体検出の実行

        Parameters
        ----------
        img_rgb : np.ndarray
            cv2.videocaptureなどから得られた画像

        Returns
        -------
        List
            List of predictions for every picture.
            Each prediction is:
            [label, confidence, xmin, ymin, xmax, ymax]
        """

        # * 2. Load some images

        self.orig_images = []  # Store the images here.
        self.input_images = []  # Store resized versions of the images here.

        for i in range(arrays.shape[0]):
            self.orig_images.append(arrays[i])
            img = Image.fromarray(arrays[i]).resize((self.img_height, self.img_width))
            img = img_to_array(img)
            self.input_images.append(img)
        
        self.input_images = np.array(self.input_images)
               
        # * 3. Make predictions
        y_pred: np.ndarray = self.model.predict_on_batch(self.input_images)  # type:ignore

        confidence_threshold = self.threshold
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                         for k in range(y_pred.shape[0])]

        return y_pred_thresh

    def results_filter(self, results: List, elements_class_dict: dict):
        """[summary]

        Parameters
        ----------
        results : List
            List of predictions for every picture.
            Each prediction is:
            [label, confidence, xmin, ymin, xmax, ymax]

        Returns
        -------
        List
            List of filtered predictions for every picture.
            Each prediction is: [label, confidence, xmin, ymin, xmax, ymax]
        """

        detected_elements: List[Element] = list()

        for i in range(len(results)):

            # 画像のdataをDataFrameで取得、confでsort
            # ! label[0] は background. elements[1] = label[0]
            df = pd.DataFrame(data=results[i], columns=self.columns)

            img_results = df.sort_values('conf', ascending=False).values
            det_label = img_results[:, 0]
            det_conf = img_results[:, 1]
            det_xmin = img_results[:, 2]
            det_ymin = img_results[:, 3]
            det_xmax = img_results[:, 4]
            det_ymax = img_results[:, 5]

            # 各要素のカウント変数を初期化
            each_element_counts = [0] * self.config.data['elem_num']

            # 各抽出要素に対する処理
            for j in range(det_conf.shape[0]):

                xmin = det_xmin[j] * \
                    self.orig_images[i].shape[1] / self.img_width
                ymin = det_ymin[j] * \
                    self.orig_images[i].shape[0] / self.img_height
                xmax = det_xmax[j] * \
                    self.orig_images[i].shape[1] / self.img_width
                ymax = det_ymax[j] * \
                    self.orig_images[i].shape[0] / self.img_height
                score = det_conf[j]

                label = int(det_label[j])
                name = self.e_names[label-1]

                each_element_counts[label-1] += 1

                # 各要素最大個数までの処理
                if each_element_counts[label-1] <= self.e_nums[label-1]:
                    detected_elements.append(Element(label=label, name=name,
                                                     od_conf=score,
                                                     xmin=xmin, ymin=ymin,
                                                     xmax=xmax, ymax=ymax,
                                                     color=self.color_codes[label-1],
                                                     classes = elements_class_dict[name],
                                                     index = i))

        return detected_elements


class ElementStateClassificator:
    """作業要素の状態の分類を行うクラス"""
    # ! 分類を行うだけであって、完了判定には使用しないこと

    def __init__(self, config: utl.Config, element_number: int):
        """elementの状態判別を行うためのモデル構築

        Parameters
        ----------
        config : Config
            コンフィグ情報をまとめている。samos_utilsで定義。
            configファイルは.json。（default_name: config.json）
        element_number : int
            モデルを構築するconfig上でのelement番号
        """
        self.element_id = element_number
        hdf5_path = Path(config.data["element"]
                         [element_number]["ic_h5"]).resolve()
        json_path = Path(config.data["element"]
                         [element_number]["ic_json"]).resolve()

        # アーキテクチャ読み込み
        json_string = open(json_path).read()
        self.state_model: tf.keras.Model = model_from_json(  # type:ignore
            json_string)
        # 重み読み込み
        self.state_model.load_weights(str(hdf5_path))   #type:ignore
        # コンパイル
        self.state_model.compile(   #type:ignore
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def classify(self, src):
        """
        推論の実行

        Args:
            src (np.darray): 入力画像

        Returns:
            np.darray: Numpy array(s) of predictions.
        """        

        return self.state_model.predict_on_batch(src)

class InputCounter:
    def __init__(self,area_roi=(70,0,160,115),line_roi=(0,10,90,11),hsv_min=(0,0,0),hsv_max=(255,255,30),cycle_threshold=450):
        
        self.area_roi = area_roi
        self.line_roi = line_roi
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max
        self.cycle_threshold = cycle_threshold

        self.cur_flag = False
        self.prev_flag = False
        self.standby = True
        
        self.count = 0
        self.wait_frame = 0
        
        self.input_nums:List[int] = []
        self.input_times:List[datetime.datetime] = []
        self.input_frame:List[int] = []
         
        self.kernel = np.ones((10,10),np.uint8)

    def input_check(self, frame, frame_num):
                
        area_roi = frame[self.area_roi[1]:self.area_roi[3],self.area_roi[0]:self.area_roi[2]]
        hsv_masked_area = self.hsv_mask(area_roi,self.hsv_min,self.hsv_max)
        closing_area = cv2.morphologyEx(hsv_masked_area,cv2.MORPH_CLOSE,self.kernel)
        
        line_roi = closing_area[self.line_roi[1]:self.line_roi[3],self.line_roi[0]:self.line_roi[2]]
        arr = line_roi
        
        self.cur_flag = np.any(arr==255)
        
        if self.cur_flag and self.standby and not self.prev_flag:
            self.new_input(frame_num)

        if not self.standby:
            self.waiting()

        # * 後処理
        self.prev_flag = self.cur_flag

    def hsv_mask(self,image, hsvLower, hsvUpper):
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 画像をHSVに変換
        hsv_masked= cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
        return hsv_masked

    def new_input(self,frame_num):
        self.count+=1
        print("input_count:",self.count)
        self.input_nums.append(self.count)
        self.input_times.append(datetime.datetime.now())
        self.input_frame.append(frame_num)
        self.wait_frame = 0
        self.standby = False

    def exist_check(self,frame,frame_num):

        area_roi = frame[self.area_roi[1]:self.area_roi[3],self.area_roi[0]:self.area_roi[2]]
        hsv_masked_area = self.hsv_mask(area_roi,self.hsv_min,self.hsv_max)
        closing_area = cv2.morphologyEx(hsv_masked_area,cv2.MORPH_CLOSE,self.kernel)
        
        contours, hierarchy = cv2.findContours(closing_area,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 1000, contours))

        if len(contours) >= 1:
            if cv2.contourArea(contours[0]) > 3000:
                self.count += 2
            else:
                self.count += 1
            self.input_nums.append(self.count)
            self.input_times.append(datetime.datetime.now())
            self.input_frame.append(frame_num)
            print("input_count:",self.count)

    def waiting(self):
        self.wait_frame += 1
        if self.wait_frame >= self.cycle_threshold:
            self.standby = True


class ImageInput():
    """画像取得用クラス"""

    def __init__(self, path: str, with_basler: bool = True,batch_size:int = 4):
        """[summary]

        Parameters
        ----------
        path : str
            ipAddres(basler camera) OR video file path
        with_cam : bool, optional
            baslerのカメラから入力する場合にはTrue, by default True

        """

        self.online = with_basler
        self.path = path
        self.suc_flag: bool = False
        self.batch_size = batch_size
        self.std_width = 1
        self.std_height = 1
        self.channels = 3
        self.frame_count = 0
        self.frame_count_batch:list[int] = list()
        self.missing_frame_count: int = 0
        self.cached_frame_count = 0
        self.now = datetime.datetime.now()
        self.time_batch:list[datetime.datetime] = list()
        self.is_cached = False
        self.accessing = False
        self.ret = True
        
        if self.online:
            print(path)
            try:
                # RTMPサーバーからストリームを取得する
                self.capture = cv2.VideoCapture(path)
                if not self.capture.isOpened():
                    tkinter.messagebox.showerror("接続不可", "RTMPサーバーに接続できませんでした。URLをご確認ください。")
                    self.ret = False
                else:
                    print("RTMPサーバーに接続しました")
                    self.std_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.std_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.ret = True
            except Exception as e:
                print(f"RTMP接続エラー: {e}")
                self.ret = False

                
        # * debug mode
        else:
            try:
                
                tkinter.messagebox.showinfo("動画選択", "解析する動画を選択してください")
                typ = [('.mp4ファイル', '*.mp4')]
                dir = os.getcwd()
                path = tkinter.filedialog.askopenfilename(filetypes=typ,
                                                        initialdir=dir)
                self.cap = cv2.VideoCapture(path)
                if self.cap.isOpened():
                    self.suc_flag, bgr_img = self.cap.read()
                    if self.suc_flag:
                        self.std_width = bgr_img.shape[1]
                        self.std_height = bgr_img.shape[0]
                    else:
                        print("動画ファイルが読み込めませんでした")
                        tkinter.messagebox.showerror("動画読込不可","動画ファイルが読み込めませんでした")
                        self.ret = False
                else:
                    print('動画ファイルにアクセスできませんでした')
                    tkinter.messagebox.showerror("動画読込不可","動画ファイルが読み込めませんでした")
                    self.ret = False
                    
            except:
                
                self.ret = False

        self.img: np.ndarray = np.zeros(
            (self.std_height, self.std_width, self.channels),dtype=np.uint8)
        self.img_batch:np.ndarray = np.zeros(
            (self.batch_size, self.std_height, self.std_width, self.channels),dtype=np.uint8)
        

    def load(self):
        """ カメラもしくは動画から画像を取得する
        
        カメラもしくは動画からself.imgに新しい画像を取得します。
        また、self.img_arraysに画像を追加します。
        self.img_arraysがbatch_size以上になった場合には、最も古い画像ものから更新します。
        """      

        # 時間取得
        self.now = datetime.datetime.now()

        # * camera onlineの場合
        if self.online:
            # 画像取得
            # * RTMPサーバーからの画像取得の場合
            if self.online:
                # 画像取得
                ret, frame = self.capture.read()
                if ret:
                    # 画像を保存 (BGR -> RGB)
                    self.img = frame[:, :, ::-1]
                    self.suc_flag = True
                else:
                    # 読み込み失敗時にデフォルトの黒い画像を設定
                    self.img = np.zeros((self.std_height, self.std_width, 3), dtype=np.uint8)
                    self.suc_flag = False

        # * camera offlineの場合
        else:
            self.suc_flag, bgr_img = self.cap.read()
            if self.suc_flag:
                self.img = bgr_img[:, :, ::-1]
            else:
                self.img = np.zeros((1, 1, 3))

        # * バッチの作成・更新
        while self.accessing:
            time.sleep(0.03)

        self.accessing = True

        if self.frame_check(self.img):
            
            # batch != fullの場合
            if self.cached_frame_count < self.batch_size:
                # img_batchへの追加
                self.img_batch[self.cached_frame_count] = self.img
                # frame_count_batchへの追加
                self.frame_count_batch.append(self.frame_count)
                # time_batchへの追加
                self.time_batch.append(self.now)

                # increment cached_frame_count, batchsizeと同じになったらis_cachedをtrueに
                self.cached_frame_count += 1
                
                if self.cached_frame_count == self.batch_size:
                    self.is_cached = True

            # batch == fullの場合
            else:
                # img_batchの更新
                self.img_batch = np.delete(self.img_batch,0,0)
                expand_img = np.expand_dims(self.img,0)
                self.img_batch = np.vstack((self.img_batch,expand_img)) #type:ignore 

                # frame_count_batchの更新
                del self.frame_count_batch[0]
                self.frame_count_batch.append(self.frame_count)
                
                # time_batchの更新
                del self.time_batch[0]
                self.time_batch.append(self.now)

        # frame枚数のカウント
        self.frame_count += 1
        self.accessing = False

    def clear_batch(self):
        """batchのクリア"""

        while self.accessing:
            time.sleep(0.001)

        self.accessing = True

        self.cached_frame_count = 0
        self.is_cached = False
        self.img_batch = np.zeros_like(self.img_batch) # type:ignore
        self.frame_count_batch.clear()
        self.time_batch.clear()
        
        self.accessing = False

    def release(self):
        """キャプチャのリリース"""        

         # if camera online (from RTMP server)
        if self.online:
            self.capture.release()
            print("RTMPサーバーの接続を解放しました")
            
        # if camera offline (from movie)
        else:
            self.cap.release()

    def frame_check(self,src:np.ndarray):
        
        # null check
        if len(src) == 0:
            self.missing_frame_count += 1
            print("missing_frame_count",self.missing_frame_count)
            return False
        #size check
        width = src.shape[1]
        height = src.shape[0]
        if height != self.std_height or width != self.std_width:
            self.missing_frame_count += 1
            print("missing_frame_count",self.missing_frame_count)
            return False
        return True

class ProgressInterface(tk.Frame):
    """解析プログラム本体"""

    def __init__(self, 
                 config: utl.Config, master=None, rec_tgt: str = None,  #type:ignore
                 with_camera:bool=True, analysis_batch:int = 4, output_speed_data:bool=False,
                 reset:bool=False, worker:str=""):
        """解析プログラム本体

        Parameters
        ----------
        config : Config
            作業情報設定ファイル
        master : [type], optional
            ウィジェットを展開するためのtk.Tk(or tk.Frame), by default None
        with_camera : bool, optional
            オンラインカメラ（Basler）を使用するか, by default True
        debug : bool, optional
            デバッグモードのON/OFF, by default False
        """

        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            tkinter.messagebox.showwarning("GPU","GPUが確認できません。\n解析が遅くなる恐れがあります。")

        print("Now initializing...")
        
        super().__init__(master)

        self.master = master #type:ignore
        self.THRESH_MSEC = 1000
        self.THRESH_STABLE = 10
        
        # # * ライセンス認証関連
        # self.CHECK_INTERVAL_MIN = 10
        
        # self.td_check_interval = datetime.timedelta(minutes=self.CHECK_INTERVAL_MIN)
        
        # self.cur_check_point_time = datetime.datetime.now()
        # self.next_check_point_time = self.cur_check_point_time + self.td_check_interval

        
        # * インスタンス変数初期化
        # 引数
        self.config = config
        self.rec_target_name = rec_tgt
        self.with_camera = with_camera
        self.batch_size = analysis_batch
        self.output_speed_data = output_speed_data
        
        # フラグ
        self.start_flag = False
        self.close_flag = False
        self.waiting_flag = False
        self.new_img_flag = False

        # config情報
        self.title:str = self.config.data["title"]
        self.e_num: int = self.config.data["elem_num"]
        self.w_num: int = self.config.data["work_num"]
        self.w_names: List[str] = self.config.data["work_name"]
        self.process: List = self.config.data['work_order']
        self.e_names, self.e_widths, self.e_heights, self.e_max_nums = self.config.get_element_info()
        self.src_movie_info: OrderedDict = self.config.data["src"]
        
        # element関係
        self.elements: List[Element] = list()
        self.element_drawers: List[ElementDrawer] = list()

        # work関係
        self.num_tasks: int = len(self.process)
        
        # 状態関係
        self.work_observer = WorkObserver(element = self.config.data["element"],
                                          work_order=self.process,
                                          threshold=config.data["threshold"],
                                          standards=config.data["standard"],
                                          config=self.config,
                                          online = self.with_camera)
        atexit.register(self.work_observer.cleanup)
        self.stable_count = 0
        self.error_occurred = False
        
        # 時間関係
        # # 一般処理系統（読込・表示・保存）
        self.prev_reload_start_time: float = 0
        self.reload_fps: float = 0
        self.td_loading:float = 0
        self.td_reloading:float = 0
        self.td_recording:float = 0
        self.td_drawing:float = 0
        self.delay_time:int = 0
        self.frame_count:int = 0
        # # 解析系統
        self.prev_classification_end_time:float = 0
        self.td_analysis:float = 0
        self.td_object_detection:float = 0
        self.td_image_classification:float = 0
        self.td_update_status: float = 0
        
        # * 表示関係
        self.task_status_strings = ['未着手', '作業中',
                                    '作業中', '完了', 'エラー']
        self.task_color_codes = [
            'gray', 'yellow', 'yellowgreen', 'limegreen', 'red'
        ]
        self.master_status_strings = ['停止', '準備中', '待機中',
                                      'OK', '注意', '警告',
                                      'システムエラー']
        self.master_color_codes = [
            'gray', 'orange', 'cyan',
            'limegreen', 'yellow', 'red',
            'purple'
        ]

        # * 出力関係
        self.log_dir = self.config.data["analysis_log_root_dir"]
        # # element_df(sublog)
        self.elem_col = ['frame_num','time','label','name',
                         'xmin','ymin','xmax','ymax','od_conf']
        for i in range(self.w_num):
            self.elem_col.append(str(i))
                
        # * work_df(sublog)
        self.work_col = pd.MultiIndex.from_product(
            iterables=[['score', 'status'], self.work_observer.task_names],
            names=['param', 'task_name'])
        
        # * product_df(main_log)
        product_df_array = [
            ["master","master", "master", "master"],
            ["時刻","台数", "状態", "作業時間"],
        ]
        tasks_each = ["状態", "作業時間"]
        for task_name in self.work_observer.task_names:
            for i in range(len(tasks_each)):
                product_df_array[0].append(task_name)
                product_df_array[1].append(tasks_each[i])
        tuples = list(zip(*product_df_array))
        self.product_col = pd.MultiIndex.from_tuples(
            tuples, names=["category", "data"])

        # # key_info(main_log)
        self.integrate_key_log_path = ""

        self.status_cache=[]

        # * 分類器の定義
        self.e_detector = ElementDetector(config=self.config)
        self.e_s_classificators: List[ElementStateClassificator] = list()
        for i in range(int(self.config.data["elem_num"])):
            self.e_s_classificators.append(
                ElementStateClassificator(config, i))

        # * インターフェース関係定義
        DEFAULT_STATUS_FONT_SIZE = 48
        DEFAULT_INDEX_FONT_SIZE = 28
        DEFAULT_NORMAL_FONT_SIZE = 18
        DEFAULT_SUB_FONT_SIZE = 14
        DEFAULT_MONITOR_FONT_SIZE = 8
        DEFAULT_TASK_NUM = 15

        if self.num_tasks > DEFAULT_TASK_NUM:
            font_scale = DEFAULT_TASK_NUM/self.num_tasks
        else:
            font_scale = 1
        status_font_size = int(DEFAULT_STATUS_FONT_SIZE*font_scale)
        index_font_size = int(DEFAULT_INDEX_FONT_SIZE*font_scale)
        normal_font_size = int(DEFAULT_NORMAL_FONT_SIZE*font_scale)

        self.font_status = tkinter.font.Font(
            self, family="Meiryo UI", size=status_font_size, weight="bold")
        self.font_index = tkinter.font.Font(
            self, family="Meiryo UI", size=index_font_size, weight="bold")
        self.font_normal = tkinter.font.Font(self, family="Meiryo UI", size=normal_font_size)
        self.font_sub = tkinter.font.Font(
            self, family="Meiryo UI", size=DEFAULT_SUB_FONT_SIZE, slant="italic")
        self.font_monitor = tkinter.font.Font(
            self, family="Meiryo UI", size=DEFAULT_MONITOR_FONT_SIZE, slant="italic")
        self.src_width:int = int(self.src_movie_info["width"])
        self.src_height: int = int(self.src_movie_info["height"])
        self.EXPAND_RATIO = 0.7
        self.vis_width = int(self.src_width*self.EXPAND_RATIO)
        self.vis_height = int(self.src_height*self.EXPAND_RATIO)

        # * ビデオキャプチャの準備
        if with_camera:
            url: str = config.data["CameraAddress"]
            # RTMPサーバーから動画を取得する際のfpsは29.97002997002997
            fps = 29.97002997002997  # TODO 暫定
            self.load_interval: float = 1000/fps  # 画面更新のインターバル[ms]
        else:
            url: str = config.data["src"]['path']
            fps = self.config.data["src"]["fps"]
            self.load_interval: float = 1000 / fps
        self.url=url
        self.capture = ImageInput(path=url, with_basler=with_camera, batch_size = self.batch_size)

        # * 各要素と分類クラスのマップ読込
        self.elem_class_dict = dict()
        for e in config.data["element"]:
            self.elem_class_dict[e["name"]] = e["classes"]

        # * pack, widgetの作成
        self.create_widget()

        if self.work_observer.ret == False or self.capture.ret == False:
            # ウィンドウを閉じる
            self.master.destroy()  # type:ignore
            if __name__ == "__main__":
                self.master.quit()
                
        else:

            # * 表示の開始
            FIGURE_RELOAD_SEC = 1.0
            self.flow_reload_interval = fps * FIGURE_RELOAD_SEC
            self.src = None
            self.reload()

            # * 解析スレッド開始
            self.thread_analysis = threading.Thread(target=self.analysis)
            self.thread_analysis.start()

            # * reset時の挙動
            if reset:
                self.worker_name_entry.insert(tkinter.END,worker) # type:ignore
                self.start()
                        
            # self.check_license()


            self.start()
            
            print("System was initialized.")

    def create_widget(self):
        """ウィジェットの作成"""

        # メニューの作成
        menubar = tk.Menu(self)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="終了", command=self.close)
        
        self.movie_frame = tk.Frame(self)
        self.status_frame = tk.Frame(self)
        self.input_frame = tk.Frame(self)
        self.control_frame = tk.Frame(self)
        
        # self.self.movie_frame = tk.Frame(self, relief="groove")
        # # self.flow_frame = tk.Frame(self,relief="groove")
        # self.self.status_frame = tk.Frame(self, relief="groove")
        # self.self.control_frame = tk.Frame(self, relief="groove")

        # viewmenu = tk.Menu(menubar, tearoff=0)
        # viewmenu.add_command(label="統計データを表示する", command=self.view_stat)

        menubar.add_cascade(label="ファイル", menu=filemenu)
        # menubar.add_cascade(label='表示', menu=viewmenu)
        self.master.config(menu=menubar)    # type: ignore

        # 動画用の作成
        self.window_margin = 10
        self.movie_canvas = tk.Canvas(self.movie_frame,
                                width=self.vis_width,
                                height=self.vis_height,
                                background="white")
        self.movie_canvas.grid(row=0, column=0,sticky=tk.NSEW)

        # master_statusの表示
        self.strval_master_status = tk.StringVar(
            self.status_frame, self.master_status_strings[self.work_observer.master_status])
        self.lbl_master_status = ttk.Label(self.status_frame,
                                           textvariable=self.strval_master_status,
                                           background=self.master_color_codes[self.work_observer.master_status],
                                           font=self.font_status,
                                           anchor='center',
                                           relief='groove')
        self.lbl_master_status.grid(
            row=0, column=0, columnspan=3,
            padx = 5, pady=10, sticky=tk.NSEW)

        # 作業一覧の描画
        lbl_work_title = ttk.Label(self.status_frame, text='作業', font=self.font_index)
        lbl_work_title.grid(row=1, column=0, padx=10, pady=10, sticky=tk.NSEW)

        for i, p in enumerate(self.process):
            column_offset = 0 if i < 40 else 3  # 40番目までを1列目と2列目に、それ以降を新たな列に表示
            row = i if i < 40 else i - 40
            work_name_text = str(i+1) + ". " + p["work_name"]
            lbl_works = ttk.Label(
                self.status_frame, text=work_name_text, font=self.font_normal)
            lbl_works.grid(row=2+row, column=0 + column_offset, sticky=tk.NSEW, padx=10, pady=5)

        # 状態一覧の描画
        lbl_work_status = ttk.Label(self.status_frame, text='状態', font=self.font_index)
        lbl_work_status.grid(row=1, column=1, padx=10, pady=10, sticky=tk.NSEW)

        self.strings_work_status: List[tk.StringVar] = list()
        self.labels_work_status: List[ttk.Label] = list()
        self.btn_error_record: List[tk.Button] = list()
        for i in range(self.num_tasks):
            column_offset = 0 if i < 40 else 3  # 40番目までを1列目と2列目に、それ以降を新たな列に表示
            row = i if i < 40 else i - 40
            self.strings_work_status.append(tk.StringVar(self.status_frame, "未着手"))
            self.labels_work_status.append(ttk.Label(self.status_frame,
                                                    textvariable=self.strings_work_status[i],
                                                    background='#FFFFFF',
                                                    font=self.font_normal,
                                                    anchor='center'))
            self.labels_work_status[i].grid(
                row=2+row, column=1 + column_offset, 
                padx=10, pady=3, sticky=tk.NSEW
            )
            self.btn_error_record.append(tk.Button(
                self.status_frame,
                text="完了",
                state=tk.DISABLED, 
                font = self.font_normal,
                command=partial(self.error_record,i)))
            self.btn_error_record[i].grid(
                row=2+row, column=2 + column_offset, 
                padx=10, pady=3, sticky=tk.NSEW
            )


            

        # 台数表示
        self.strval_work_count = tk.StringVar(self.status_frame, '0')
        lbl_work_count = ttk.Label(
            self.status_frame, textvariable=self.strval_work_count, font=self.font_normal)
        lbl_work_count.grid(
            row=self.num_tasks+2, column=0, padx=10, sticky=tk.W)

        # 監督者モード切替
        self.admin_flag = tk.BooleanVar(self.input_frame)
        admin_check = tk.Checkbutton(self.input_frame,
                                     text="管理者モード",
                                     font=self.font_normal,
                                     variable=self.admin_flag)
        admin_check.grid(row=0,column=0,padx=5,pady=5)

        # 作業者名入力ボックスの作成
        lbl_worker_name = ttk.Label(self.input_frame, text="作業者：", font=self.font_normal)
        lbl_worker_name.grid(row=0, column=1, padx= 10, sticky=tk.E)
        self.worker_name_entry = ttk.Entry(self.input_frame,font=self.font_normal,width=10)
        self.worker_name_entry.grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.W+tk.E)

        # ボタンの作成
        self.start_button = tk.Button(
            self.control_frame, text="開始", command=self.start, 
            state=tk.NORMAL, font = self.font_index,relief=tk.RAISED,bd=8)
        self.start_button.grid(
            row=0, column=0, padx=5,pady=5,sticky=tk.NSEW)
        self.stop_button = tk.Button(
            self.control_frame, text="停止", command=self.stop, 
            state=tk.DISABLED,font = self.font_index,relief=tk.RAISED,bd=8)
        self.stop_button.grid(
            row=0, column=1, padx=5,pady=5,sticky=tk.NSEW)
        self.reset_button = tk.Button(
            self.control_frame, text="レシピ変更", command=self.reset,
            state=tk.NORMAL,font = self.font_normal,relief=tk.RAISED,bd=8)
        self.reset_button.grid(
            row=0, column=2, padx=5,pady=5,sticky=tk.NSEW)

        # * 全体のgrid
        
        # Frame自身もトップレベルウィジェットに配置
        self.grid(column=0, row=0, sticky=tk.NSEW)
        
        # self.status_frameの引き伸ばし設定
        columns = 4
        for i in range(columns+1):
            self.columnconfigure(i,weight=1)
        rows = self.num_tasks+3
        for i in range(rows):
            self.rowconfigure(i,weight=1)
            
        self.movie_frame.grid(row=0,column=0)
        self.status_frame.grid(row=0,column=1)
        self.input_frame.grid(row=1,column=0,sticky=tk.W)
        self.control_frame.grid(row=1,column=1)

        # トップレベルのウィジェットも引き伸ばしに対応させる
        self.master.columnconfigure(0,weight=1)
        self.master.rowconfigure(0,weight=1)

    # * コマンド関係
    def start(self):
        """
        作業推定の開始
        """        

        print('Now preparing the analysis...')

        # io.start()# ! comment for dev

        # widget関係
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.worker_name = self.worker_name_entry.get()
        self.work_observer.worker = self.worker_name

        # * 出力用データの準備
        dt_now = datetime.datetime.now()
        self.dst_basename = self.title + "_" + dt_now.strftime('%y%m%d_%H%M%S')
        
        # * log_dirの作成
        # ! レシピの書き換えが済み次第，絶対パス指定からレシピ指定に変更すること
        # self.log_dir = self.config.data["analysis_log_root_dir"]+"/"+dt_now.strftime('%Y-%m-%d')+"/"+self.config.data["title"]
        self.log_dir = "D:/SAMoS/log"+"/"+dt_now.strftime('%Y-%m-%d')+"/"+self.config.data["title"]
        os.makedirs(self.log_dir,exist_ok=True)
        key_dir = self.log_dir
        os.makedirs(key_dir,exist_ok=True)

        # ビデオライタの準備
        movie_log_dir = self.log_dir + "/rec"
        os.makedirs(movie_log_dir,exist_ok=True)
        dst_movie_path = movie_log_dir + "/rec_" + self.dst_basename + ".mp4"
        format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        if self.with_camera:
            # TODO camera情報を読み取ってテストすること
            # RTMPサーバーから動画を取得する際のfpsは29.97002997002997
            fps = 29.97002997002997    # 暫定
        else:
            fps = self.config.data["src"]["fps"]
        size = (self.src_width,self.src_height)
        self.writer = cv2.VideoWriter(dst_movie_path, format, fps, size)

        # list初期化
        self.elem_id = 0
        self.elem_data_log = []
        self.work_data_log = []
        self.prod_data_log = []
        self.error_data_log = []
        if self.output_speed_data:
            self.speed_data_log = []

        # dataframe初期化
        self.elem_df = pd.DataFrame(columns=self.elem_col)
        self.elem_df_id = 0
        self.work_df = pd.DataFrame(columns=self.work_col)
        self.product_df = pd.DataFrame(columns=self.product_col)

        # for dataframes and rec sync
        self.capture.frame_count = 0

        # key info 出力
        self.integrate_key_log_path = key_dir + "/key_" + self.dst_basename + ".csv"
        with open(self.integrate_key_log_path,mode='w',encoding='utf_8') as f:
            worker_name = self.worker_name_entry.get()
            f.write("worker,"+worker_name+'\n')
            f.write("product,"+self.title+'\n')
            f.write("start,"+dt_now.strftime('%Y-%m-%d %H:%M:%S.%f')+'\n')
        
        print(self.integrate_key_log_path, "has been made.")

        # * flow_df用データ書き込み
        self.work_observer.proc_start_nums.append(0)
        self.work_observer.proc_start_frame.append(0)
        self.work_observer.proc_start_time.append(datetime.datetime.now())

        self.start_flag = True
        self.work_observer.master_status = 1
        
        if self.work_observer.ret == False or self.capture.ret == False:
            print("error")
            # ウィンドウを閉じる
            self.master.destroy()  # type:ignore
            if __name__ == "__main__":
                self.master.quit()

        
    def stop(self):
        """
        作業推定の停止
        """        
        # io.stop()# ! comment for dev

        # * ストップ時の作業のtime_log出力
        statuses = self.work_observer.statuses.tolist()
        start_times = self.work_observer.task_start_time
        finish_times = self.work_observer.task_finish_time
        durations = self.work_observer.task_duration
        start_frames = self.work_observer.task_start_frame
        finish_frames = self.work_observer.task_finish_frame
        st_log = []
        ft_log = []
        d_log = []
        sf_log = []
        ff_log = []

        for s, st, ft, d, sf, ff in zip (
            statuses, start_times, finish_times, durations, start_frames, finish_frames):
            if s == 3:
                st_log.append(st)
                ft_log.append(ft)
                d_log.append(d)
                sf_log.append(sf)
                ff_log.append(ff)

        self.work_observer.times_cache.append(
            [st_log,ft_log,d_log,sf_log, ff_log]
        )

        # * 各種データリセット
        self.start_flag = False
        self.work_observer.master_status = 0
        self.work_observer.work_count = 0
        self.work_observer.statuses[:] = 0
        self.work_observer.scores[:]=0

        # widget関係
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.movie_canvas.delete('elem')

        while(self.waiting_flag):
            time.sleep(1.)

        time.sleep(1)

        # * ファイル出力
        self.output_analysis_logs()
        
        print('Now waiting...')

    def reset(self,new_recipe_name:str="unknown"):

        print('reset start')

        self.close_flag = True

        self.work_observer.mm.close()

        # 解析終了待機、解析実行中なら終了処理
        self.thread_analysis.join()

        # * 中断処理
        if self.start_flag:
            self.stop()

        # 各種コンストラクタを開放
        self.capture.release()

        # * 再初期化
        config = utl.Config()
        
        if self.with_camera and new_recipe_name != "unknown":
            cur_line = "04"
            target_recipe_path = "./_model/" + cur_line + "/" + new_recipe_name + '/' + new_recipe_name + '.json'
            print(target_recipe_path)
            config_path = config.load_data(target_recipe_path)
        else:
            cur_line = "04"
            target_recipe_path = "./_model/" + cur_line + "/_/_.json"
            print(target_recipe_path)
            config_path = config.load_data()
        
        self.worker_name = self.worker_name_entry.get()
        
        if config_path != "":
            print("解析モードで再起動します")
            prog_interface = ProgressInterface(config=config, 
                                               master=self.master,
                                               rec_tgt=self.rec_target_name,
                                               with_camera=self.with_camera,
                                               analysis_batch=self.batch_size, 
                                               output_speed_data=self.output_speed_data,
                                               reset=True,
                                               worker=self.worker_name)
        else:
            print("録画モードで再起動します")
            recorder = Recorder(master=self.master,
                                product_name=new_recipe_name,
                                with_camera=self.with_camera,
                                camera_url=self.url)

    def job_complete(self,job_id:int):
        """
        作業を強制的に完了させる関数

        Args:
            job_id (int): 対象となる作業の番号
        """        

        self.work_observer.force_complete(job_id,self.frame_count)

    def job_cancel(self,label:int):
        """
        作業を強制的にキャンセルする関数

        Args:
            label (int): 対象となる作業の番号
        """        
        self.work_observer.force_cancel(label,self.frame_count)

    def error_record(self,label:int):
        
        def error_report():
            
            error_code=self.error_code.get()
            
            now = datetime.datetime.now()
            error_que = [now,self.frame_count,self.work_observer.work_count,label,error_code]
            self.error_data_log.append(error_que)
            
            # * エラー登録後の処理
            # エラー発生作業を完了状態に
            self.work_observer.force_complete(label,self.frame_count)
            # 通報ボタンをDISABLEに
            self.btn_error_record[label].config(state=tk.DISABLED)
            # ポップアップを破棄
            self.popup_error_report.destroy()
            
        def error_cancel():
            self.popup_error_report.destroy()
        
        self.popup_error_report = tk.Toplevel()
        self.popup_error_report.title("エラー報告")
        
        self.error_code = tk.IntVar(value=1)
        
        text_error_report = tk.Label(self.popup_error_report,
                                     text="エラーの原因を選択してください",
                                     font=self.font_index)
        
        radio_program_error = tk.Radiobutton(self.popup_error_report,
                                             text="正しく作業をしたが完了にならない",
                                             variable=self.error_code,
                                             font=self.font_index,
                                             value=1)
        radio_wrong_order = tk.Radiobutton(self.popup_error_report,
                                             text="誤った作業手順で作業を行った",
                                             variable=self.error_code,
                                             font=self.font_index,
                                             value=2)
        radio_wrong_process = tk.Radiobutton(self.popup_error_report,
                                             text="通常と異なる方法で作業を行った",
                                             variable=self.error_code,
                                             font=self.font_index,
                                             value=3)
        report_button = tk.Button(self.popup_error_report,
                                  text = "報告",
                                  font=self.font_index,
                                  command=error_report)
        cancel_button = tk.Button(self.popup_error_report,
                                  text = "キャンセル",
                                  font=self.font_index,
                                  command=error_cancel)
        
        text_error_report.grid(row=0,column=0,columnspan=2,padx=5,pady=5,sticky=tk.W)
        radio_program_error.grid(row=1,column=0,columnspan=2,padx=5,pady=10,sticky=tk.W)
        radio_wrong_order.grid(row=2,column=0,columnspan=2,padx=5,pady=10,sticky=tk.W)
        radio_wrong_process.grid(row=3,column=0,columnspan=2,padx=5,pady=10,sticky=tk.W)
        report_button.grid(row=4,column=0,padx=5,pady=10)
        cancel_button.grid(row=4,column=1,padx=5,pady=10)

    def output_analysis_logs(self):
        """
        各種解析用ログの出力
        """        
        
        config_path = self.config.data["config_default_path"]
        abs_config_path = Path.absolute(Path(config_path))
        config_path = str(abs_config_path)
        
        movie_log_dir = self.log_dir + "/rec"
        os.makedirs(movie_log_dir,exist_ok=True)
        movie_log_path = movie_log_dir + "/rec_" + self.dst_basename + ".mp4"
        
        elem_log_dir = self.log_dir + "/element"
        os.makedirs(elem_log_dir,exist_ok=True)
        elem_log_path = elem_log_dir + "/element_" + self.dst_basename + ".csv"

        work_log_dir = self.log_dir + "/work"
        os.makedirs(work_log_dir,exist_ok=True)
        work_log_path = work_log_dir + "/work_" + self.dst_basename + ".csv"
        
        product_log_dir = self.log_dir + "/product"
        os.makedirs(product_log_dir,exist_ok=True)
        product_log_path = product_log_dir + "/product_" + self.dst_basename + ".csv"
        
        flow_log_dir = self.log_dir + "/flow"
        os.makedirs(flow_log_dir,exist_ok=True)
        flow_log_path = flow_log_dir + "/flow_" + self.dst_basename + ".csv"
        
        time_log_dir = self.log_dir + "/time"
        os.makedirs(time_log_dir,exist_ok=True)
        time_log_path = time_log_dir + "/time_" + self.dst_basename + ".csv"
        
        status_log_dir = self.log_dir + "/status"
        os.makedirs(status_log_dir,exist_ok=True)
        status_log_path = status_log_dir + "/status_" + self.dst_basename + ".csv"

        error_log_dir = self.log_dir + "/error"
        os.makedirs(error_log_dir,exist_ok=True)
        error_log_path = error_log_dir + "/error_" + self.dst_basename + ".csv"
        
        # key書き込み
        with open(self.integrate_key_log_path,mode='a',encoding='utf_8') as f:
            f.write('end,'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+'\n')
            f.write('config_path,'+config_path+'\n')
            f.write('movie_path,'+movie_log_path+'\n')
            f.write('element_log_path,'+elem_log_path+'\n')
            f.write('work_log_path,'+work_log_path+'\n')
            f.write('product_log_path,'+product_log_path+'\n')
            f.write('flow_log_path,'+flow_log_path+'\n')
            f.write('time_log_path,'+time_log_path+'\n')
            f.write('status_log_path,'+status_log_path+'\n')
            f.write('error_log_path,'+error_log_path+'\n')
        print(self.integrate_key_log_path, "has been updated.")
        
        # 各種log 出力
        self.output_rec(movie_log_path)
        self.output_elem_log(elem_log_path)
        # self.output_work_log(work_log_path)
        # self.output_product_log(product_log_path)
        self.output_flow_data(flow_log_path)
        self.output_time_data(time_log_path)
        self.output_status_data(status_log_path)
        self.output_error_data(error_log_path)

        
        if self.output_speed_data:
            speed_log_path = "./syslog/speed_" + self.dst_basename + ".csv"
            self.output_speed_log(speed_log_path)

        # dataのクリア
        self.status_cache = []
        self.frame_count = 0
        self.work_observer.proc_start_time = []
        self.work_observer.proc_start_frame = []
        self.work_observer.proc_start_nums = []
        self.work_observer.proc_comp_time = []
        self.work_observer.proc_comp_nums = []
        self.work_observer.proc_comp_frame = []
        self.work_observer.times_cache = []
        
        
    def output_rec(self,dst):
        """
        動画の保存

        Args:
            dst (str): 出力先のパス
        """        
        self.writer.release()
        print(dst, "has been output.")

    def output_elem_log(self,dst):
        """
        検出対象推定結果のログ出力

        Args:
            dst (str): 出力先のパス
        """        
        
        # print("検出対象要素のログの作成中です")
        self.elem_df=pd.DataFrame(self.elem_data_log)
        
        self.elem_df.to_csv(dst, encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_work_log(self,dst):
        """
        作業推定結果のログ出力

        Args:
            dst (str): 出力先のパス
        """        

        # print("作業のログの作成中です")

        self.work_df = pd.DataFrame(self.work_data_log)

        self.work_df.to_csv(dst, encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_product_log(self,dst):
        """
        各種作業・推定結果のパス

        Args:
            dst (str): 出力先のパス
        """        

        # print("成果のログの作成中です")
        print(self.prod_data_log)

        self.product_df = pd.DataFrame(self.prod_data_log)

        self.product_df.to_csv(dst, encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_speed_log(self,dst):
        
        speed_col = [
            'frame', 'Reload FPS', 'Reload Time', 
            'loading', 'recording', 'drawing', 'delay',
            'Analysis FPS', 'Analysis Time',
            'detection','classification', 'update'
        ]
        speed_df = pd.DataFrame(data=self.speed_data_log, columns=speed_col)
        speed_df = speed_df.set_index('frame')
        speed_df.to_csv(dst, encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_flow_data(self,dst):

        input_x = self.work_observer.proc_start_time
        input_y = self.work_observer.proc_start_nums
        input_f = self.work_observer.proc_start_frame
        input_df = pd.DataFrame({
            'input_x': input_x,
            'input_y': input_y,
            'input_f': input_f
        })
        
        output_x = self.work_observer.proc_comp_time
        output_y = self.work_observer.proc_comp_nums
        output_f = self.work_observer.proc_comp_frame
        output_df = pd.DataFrame({
            'output_x':output_x,
            'output_y':output_y,
            'output_f':output_f
        })

        dst_flow_df = pd.concat([input_df,output_df],axis=1)
        dst_flow_df['unit'] = dst_flow_df['input_y']
        dst_flow_df = dst_flow_df.set_index('unit')

        dst_flow_df.to_csv(dst, encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_time_data(self,dst):
        
        # i台目ごとのデータを抜き取り
        time_data = []

        for i, c in enumerate(self.work_observer.times_cache):
            # w作業毎に抜き取り
            for w, (s,f,d,sf,ff) in enumerate(zip(c[0],c[1],c[2],c[3],c[4])):
                if w == 0:
                    time_data.append([i+1,w,s,s,0,sf,ff])
                time_data.append([i+1,w+1,s,f,d,sf,ff])
                    

        df_time = pd.DataFrame(
            time_data, 
            columns=["unit","work","start","finish","duration","start_frame","finish_frame"])
        df_time.to_csv(dst,encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_status_data(self,dst):
        
        status_data = []
        for c in self.status_cache:
            status_data.append([c[0],c[1],c[2],c[3],c[4]])
           
        df_status = pd.DataFrame(
            status_data, 
            columns=["frame","time","cycle","master_status","statuses"])
        df_status.to_csv(dst,encoding="utf_8_sig")
        print(dst, "has been output.")

    def output_error_data(self,dst):

        df_error = pd.DataFrame(
            self.error_data_log,
            columns=["time","frame","cycle","task_id","error_type"]
        )
        df_error.to_csv(dst,encoding="utf_8_sig")
        print(dst,"has been output.")

    def close(self):
        """終了処理を行う"""
        print('Now finalizing...')

        self.close_flag = True

        self.work_observer.mm.close()

        # 解析終了待機、解析実行中なら終了処理
        self.thread_analysis.join()
        if self.start_flag:
            self.stop()

        # 各種コンストラクタを開放
        self.capture.release()

        # ウィンドウを閉じる
        self.master.destroy()  # type:ignore

        if __name__ == "__main__":
            self.master.quit()

        # * メモリ解放
        gc.collect()
        tf.keras.backend.clear_session()

        print('Finalized.\n')

        if __name__ == "__main__":
            exit()

    def error_responce(self):
        # * エラー発生時に走る関数です
        print("エラーが発生しました")
        # io.buzzer_on()# ! comment for dev

    def error_resolution(self):
        # * エラー解消時に走る関数です
        if self.error_occurred:
            print("エラーを解決しました")
        # io.buzzer_off()# ! comment for dev

    # def check_license(self):
    #     """ライセンスの認証"""
    #     print("ライセンス認証を行います")
        
    #     bool_license = True # temp
        
    #     # TODO ライセンス承認用プログラムの記述（～次のTODOまで）
        
    #     # TODO
        
    #     if bool_license:
    #         print("ライセンスが認証されました")
    #     else:
    #         tkinter.messagebox.showwarning("ライセンス認証","ライセンスが認証されませんでした\nプログラムを終了します")
    #         self.close()

    # * コマンド　ここまで

    # * 定期実行系
    def reload(self):
        """定期的な表示の更新"""

        if not self.close_flag:

            reload_start_time = time.perf_counter()
            
            # captureの更新
            self.capture.load()

            # 画像取得状態の確認
            success_flag = self.capture.suc_flag
            if success_flag:
                pass
            else:
                self.stop()
                tkinter.messagebox.showinfo(
                    "中断", '入力が途絶えました。\n プログラムを終了します。')
                self.close()
                return

            # 画像取得と変換
            self.src = self.capture.img
            resized_img = cv2.resize(
                    self.src, (self.vis_width, self.vis_height))
            img_pil = utl.arr_to_pil(resized_img, False)
            img_itk = utl.pil_to_itk(img_pil=img_pil, master=self.movie_canvas)

            # 画像表示
            self.movie_canvas.delete("fig")
            self.movie_canvas.photo = img_itk  # type:ignore
            self.movie_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.movie_canvas.photo, tags='fig')  # type:ignore

            # 作業情報更新
            for i, status in enumerate(self.work_observer.statuses.tolist()):
                self.strings_work_status[i].set(
                    self.task_status_strings[int(status)])
                self.labels_work_status[i].configure(
                    background=self.task_color_codes[int(status)])

            self.strval_work_count.set("台数："+str(self.work_observer.work_count))

            # master_status_widgetの更新
            self.strval_master_status.set(
                self.master_status_strings[self.work_observer.master_status])
            self.lbl_master_status.configure(
                background=self.master_color_codes[self.work_observer.master_status])

            record_start_time = time.perf_counter()

            # * start時
            if self.start_flag:

                # * state_logger
                cur_states = []
                cur_states.append(self.frame_count)
                cur_states.append(datetime.datetime.now())
                cur_states.append(self.work_observer.work_count)
                cur_states.append(self.work_observer.master_status)
                cur_states.append(self.work_observer.statuses.tolist())
                self.status_cache.append(cur_states)

                # * 録画用image取得
                dst = cv2.cvtColor(self.src, cv2.COLOR_RGB2BGR)
                self.writer.write(dst)
                self.frame_count+=1

            record_end_time = time.perf_counter()

            #  検出対象表示
            self.movie_canvas.delete('elem')
            if self.start_flag and \
                len(self.element_drawers) and \
                    self.work_observer.master_status >= 2: 
                for ed in self.element_drawers:
                    ed.draw(self.movie_canvas)
            #  解析速度表示
            self.draw_analysis_fps()
            
            # 解析速度出力
            if self.output_speed_data:
                if self.td_analysis != 0 and self.start_flag:
                    self.append_speed_log()
            
            reload_end_time = time.perf_counter()

            # * 所要時間計算
            self.td_reloading = reload_end_time-reload_start_time
            self.td_loading =  record_start_time - reload_start_time
            self.td_recording = record_end_time-record_start_time
            self.td_drawing = reload_end_time-record_end_time
            self.reload_fps = 1 / (reload_start_time-self.prev_reload_start_time)

            self.prev_reload_start_time = reload_start_time

            # * エラー発生時の対応
            if not self.error_occurred and self.work_observer.master_status >= 4:
                self.error_occurred = True
                self.error_responce()
                
            response_mask:list[bool] = (self.work_observer.statuses != 3)
            
            # 管理者モードでの対応
            if self.admin_flag.get():
                for i, k in enumerate(response_mask):
                    if k:
                        self.btn_error_record[i].config(state=tk.NORMAL)
                    else:
                        self.btn_error_record[i].config(state=tk.DISABLED)
            else:
                for b in self.btn_error_record:
                    b.config(state=tk.DISABLED)
            
            
            # エラー解消時
            if not np.any(self.work_observer.statuses == 4):
                self.error_resolution()
                self.error_occurred = False
        
            # * レシピ変更の検出
            if self.work_observer.mm[2000] == 1:
                new_recipe_name = self.work_observer.mm[2004:2515].decode('sjis', 'ignore').replace('\0', '')
                print("recipe changed to " + new_recipe_name + "...")
                self.work_observer.mm[2000] = 0
                self.reset(new_recipe_name)
                
            # # * ライセンス認証
            # current_time = datetime.datetime.now()
            # if current_time > self.next_check_point_time:
            #     self.check_license()
            #     self.next_check_point_time = current_time + self.td_check_interval

            # * Recall
            self.delay_time = round(self.load_interval - self.td_reloading*1000)
            if self.delay_time < 0:
                self.delay_time = 0

            if not self.close_flag: # 処理中に中断された場合のためのフェイルセーフ
                self.after(self.delay_time, self.reload)

    def draw_analysis_fps(self):
        """時間情報の描画"""

        # 'elem'タグの要素削除
        self.movie_canvas.delete('time_info')

        text: list[str] = []
        text.append("Reload FPS = {:.1f}".format(self.reload_fps))
        text.append("Reload Time= {:>05.1f} ms".format(self.td_reloading*1000))
        text.append("\tloading time =\t{:>05.1f} ms".format(self.td_loading*1000))
        text.append("\trecording time =\t{:>05.1f} ms".format(self.td_recording*1000))
        text.append("\tdrawing time =\t{:>05.1f} ms".format(self.td_drawing*1000))
        text.append("\tdelay time =\t{:>6d} ms".format(self.delay_time))

        if self.td_analysis == 0 or not self.start_flag:
            text.append("Analysis FPS = None")
            text.append("Analysis Time= None")
            text.append("\tobject detection time =\tNone")
            text.append("\timage classification time =\tNone")
            text.append("\tupdate status time =\tNone")
        else:
            analysis_fps = 1/self.td_analysis*self.batch_size
            text.append("Analysis FPS = {:.1f}".format(analysis_fps))
            text.append("Analysis Time= {:.1f} ms".format(self.td_analysis*1000))
            text.append("\tobject detection time =\t{:>06.1f} ms".format(self.td_object_detection*1000))
            text.append("\timage classification time =\t{:>06.1f} ms".format(self.td_image_classification*1000))
            text.append("\tupdate status time =\t{:>06.1f} ms".format(self.td_update_status*1000))


        # テキスト描画
        for i,t in enumerate(text):
            self.movie_canvas.create_text(
                15, 15+15*i,
                text=t, anchor = 'sw',
                font=self.font_monitor,
                fill = '#FF0000',
                tags = 'time_info'
            )

    def append_speed_log(self):

        cur_speed = [
            self.capture.frame_count,
            self.reload_fps,
            self.td_reloading*1000,
            self.td_loading*1000,
            self.td_recording*1000, 
            self.td_drawing*1000, 
            self.delay_time,
            1/self.td_analysis*self.batch_size, 
            self.td_analysis*1000,
            self.td_object_detection*1000, 
            self.td_image_classification*1000,
            self.td_update_status*1000
        ]
        
        self.speed_data_log.append(cur_speed)        

    def append_work_data(self, frame_num: int, time:datetime.datetime):
        """productログへの記入"""

        cur_prod_s = pd.Series(index=self.product_col,
                               name=frame_num,
                               dtype=object)
        cur_work_s = pd.Series(index=self.work_col,
                               name=frame_num,
                               dtype='float64')

        temp_d = self.work_observer.process_duration
        temp_s = self.master_status_strings[self.work_observer.master_status]
        # cur_prod_s["master", "工程"] = self.title
        cur_prod_s["master", "時刻"] = time.strftime('%Y-%m-%d %H:%M:%S.%f')
        cur_prod_s["master", "台数"] = self.work_observer.work_count
        cur_prod_s["master", "状態"] = temp_s
        cur_prod_s["master", "作業時間"] = temp_d

        for task_name, task_duration, status in zip(self.work_observer.task_names,
                                                    self.work_observer.task_duration,
                                                    self.work_observer.statuses.tolist()):
            cur_prod_s[task_name, "状態"] = self.task_status_strings[status]
            cur_prod_s[task_name, "作業時間"] = task_duration
            cur_work_s['score', task_name] = task_duration
            cur_work_s['status', task_name] = status

        # * [work|product]_data_logにappend
        self.work_data_log.append(cur_work_s)
        self.prod_data_log.append(cur_prod_s)

    # * 定期実行系　ここまで

    def analysis(self):
        """解析コア"""

        # close_flagが立つまでループ
        while not self.close_flag:

            # start_flagが立ったら開始
            if self.start_flag:
                
                self.waiting_flag = False
                escape_flag = False

                # * 主な処理はここに書き加えていく

                # captureのバッチが溜まるまで待機
                if self.capture.is_cached:
                
                    # capture
                    times = copy.deepcopy(self.capture.time_batch)
                    frames = copy.deepcopy(self.capture.frame_count_batch)
                    src_imgs = copy.deepcopy(self.capture.img_batch)

                    if len(src_imgs) == 0: # イメージが入っていなかった場合の逃げ
                        escape_flag = True
                    if escape_flag:
                        print("len(src_img)=")
                        continue

                    self.capture.clear_batch()

                    # チェックポイント
                    analysis_start_time = time.perf_counter()

                    # * 物体検出
                    dtc_results = list()
                    dtc_results = self.e_detector.detection(arrays=src_imgs)
                    
                    self.elements: List[Element] = self.e_detector.results_filter(
                        results=dtc_results, elements_class_dict = self.elem_class_dict)

                    # 検出が無ければcontinue
                    if len(self.elements) == 0:
                        escape_flag = True
                    if escape_flag:
                        continue

                    for e in self.elements:
                        e.frame_num = frames[e.src_img_index]
                        e.time = times[e.src_img_index]

                    # * 物体抽出（elem_crop）
                    elem_arrays = [[] for i in range(self.e_num)]
                    for e in self.elements:
                        
                        # crop
                        e_center = [(e.ymin + e.ymax)/2, (e.xmin + e.xmax)/2]
                        x1 = int(e_center[1] - self.
                        e_widths[e.label-1] / 2)
                        x2 = int(e_center[1] + self.e_widths[e.label-1] / 2)
                        y1 = int(e_center[0] - self.e_heights[e.label-1] / 2)
                        y2 = int(e_center[0] + self.e_heights[e.label-1] / 2)
                        temp_width = x2 - x1
                        temp_height = y2 - y1
                        
                        # intへの丸めで x2 - x1 != width, y2 - y1 != heightとなった場合の調整
                        if temp_width != self.e_widths[e.label-1]:
                            x2 = x2 + (self.e_widths[e.label-1] - temp_width)
                        if temp_height != self.e_heights[e.label-1]:
                            y2 = y2 + (self.e_heights[e.label-1] - temp_height)

                        if y1 == y2 or x1 == x2:
                            continue

                        if x1 < 0:
                            x1 = 0
                            x2 = self.e_widths[e.label-1]
                        if x2 > self.src_width:
                            x1 = self.src_width - self.e_widths[e.label-1]
                            x2 = self.src_width
                        if y1 < 0:
                            y1 = 0
                            y2 = self.e_heights[e.label-1]
                        if y2 > self.src_height:
                            y1 = self.src_height - self.e_heights[e.label-1]
                            y2 = self.src_height
                        
                        e_array = src_imgs[e.src_img_index][y1:y2, x1:x2]
                        
                        # 前処理 (RGB2BGR and VGG-16のRGB値の平均値を引く)
                        e_array = preprocess_input(e_array)
                        e_array = np.expand_dims(e_array, axis=0)
                        
                        # 正規化
                        e_array = e_array / 255

                        # element size check
                        height = e_array.shape[1]
                        width = e_array.shape[2]
                        if height == self.e_heights[e.label-1] and width == self.e_widths[e.label-1]:
                            elem_arrays[e.label-1].append(e_array)

                    # チェックポイント
                    object_detection_end_time = time.perf_counter()

                    # * 画像分類
                    # label毎に実行
                    ic_results = [[] for i in range(self.e_num)]
                    for label in range(self.e_num):
                        
                        ic_inputs_array = np.zeros(
                            (len(elem_arrays[label]), self.e_heights[label], self.e_widths[label], 3))
                        
                        for i in range(len(elem_arrays[label])):
                            ic_inputs_array[i] = elem_arrays[label][i]
                        
                        ic_results[label] = self.e_s_classificators[label].classify(src=ic_inputs_array) # type:ignore
                        
                    # * 推論結果をElementに記入
                    for label in range(self.e_num):
                        i = 0
                        for e in self.elements:
                            if label == e.label-1:
                                try:
                                    e.ic_conf = ic_results[label][i]
                                except IndexError as e:
                                    print("* Handring IndexError")
                                    print('label:',label,'i:',i)
                                    escape_flag = True
                                else:
                                    i += 1
                            if escape_flag:
                                break
                        if escape_flag:
                            break
                    if escape_flag:
                        continue

                    # * s_elem:pd.Seriesを作成し、elements_logへ追記
                    drawn_elem = []
                    self.element_drawers = []
                    for e in self.elements:
                        
                        # s_base（基礎情報：label,name,）の作成
                        s_base_val = [e.frame_num, e.time, e.label, e.name,
                                      e.xmin, e.ymin, e.xmax, e.ymax, e.od_conf]
                        s_base_index = ['frame_num','time','label','name',
                                        'xmin','ymin','xmax','ymax','od_conf']
                        s_base = pd.Series(s_base_val, index=s_base_index)
                        
                        # s_classifyの作成
                        s_classify = pd.Series(e.ic_conf, index=e.classes)
                        
                        # s_baseとs_classifyの結合
                        s_elem = pd.concat([s_base, s_classify])

                        # dict変換
                        elem_dict = s_elem.to_dict()

                        # * elem_data_log への追記
                        self.elem_data_log.append(elem_dict)

                        self.elem_df_id += 1

                        # * 描画用
                        if not (e.label in drawn_elem):
                            
                            self.element_drawers.append(
                                ElementDrawer(
                                    element=e, 
                                    expand_ratio=self.EXPAND_RATIO,
                                    work_names = self.w_names,
                                    font = self.font_sub
                                )
                            )
                            drawn_elem.append(e.label)

                    # チェックポイント
                    classification_end_time = time.perf_counter()    

                    # * 状態判定
                    if self.prev_classification_end_time != 0:
                        
                        self.td_analysis = (classification_end_time - analysis_start_time)
                        analysis_time_ratio = self.td_analysis / self.batch_size

                        classification_end_time = time.perf_counter()
                        # フレームごとにまとめて処理
                        for i in range(self.batch_size):
                            elements_in_frame = []
                            for e in self.elements:
                                if i == e.src_img_index:
                                    elements_in_frame.append(e)

                            if len(elements_in_frame) > 0:

                                # score加算
                                self.work_observer.update_scores(elements_in_frame,analysis_time_ratio)

                            #     # 状態の更新
                                self.work_observer.update_states(self.frame_count)
                                
                                # product_df & work_dfへの追記
                                self.append_work_data(frame_num=frames[i], time=times[i])

                    # チェックポイント
                    update_status_end_time = time.perf_counter()

                    # * 処理時間計算
                    self.td_object_detection = object_detection_end_time - analysis_start_time
                    self.td_image_classification = classification_end_time - object_detection_end_time
                    self.td_update_status = update_status_end_time - classification_end_time
                    self.prev_classification_end_time = classification_end_time

                    # * 解析時間がTHRESH_MSEC以下なら待機状態に移行
                    if self.work_observer.master_status == 1:
                        if self.td_analysis < self.THRESH_MSEC/1000:
                            self.stable_count += 1
                        else:
                            self.stable_count = 0
                        
                        if self.stable_count >= self.THRESH_STABLE:
                            print("Analysis start.")

                            self.work_observer.master_status = 2

                else:
                    time.sleep(0.03)

            else:
                # 待機中
                self.waiting_flag = True
                time.sleep(1.)

class Recorder(tk.Frame):
    """録画モードプログラム本体"""

    def __init__(self,
                 master=None,
                 product_name:str="unknown",
                 with_camera:bool=True,
                 camera_url:str="192.168.230.11"):
        """録画モードプログラム本体"""

        print("Recording Mode")

        super().__init__(master)
        self.with_camera = with_camera
        # GoProから動画を取得する際のサイズは1280*720
        WIDTH = 1280
        HEIGHT = 720
        # RTMPサーバーから動画を取得する際のfpsは29.97002997002997
        FPS = 29.97002997002997
        BATCH_SIZE = 1

        log_dir = "D:/SAMoS/new_product_movie"
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        movie_log_dir = log_dir + "/" + product_name + "/"
        os.makedirs(movie_log_dir,exist_ok=True)
        dst_movie_path = movie_log_dir +"/rec_"+ product_name + "_" + now_str + ".mp4"
        
        format = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        # * インターフェース関係定義
        self.font_status = tkinter.font.Font(
            self, family="Meiryo UI", size=48, weight="bold")
        self.font_index = tkinter.font.Font(
            self, family="Meiryo UI", size=28, weight="bold")
        font_normal_size = 18
        self.font_normal = tkinter.font.Font(self, family="Meiryo UI", size=font_normal_size)
        self.font_sub = tkinter.font.Font(
            self, family="Meiryo UI", size=14, slant="italic")
        self.font_monitor = tkinter.font.Font(
            self, family="Meiryo UI", size=8, slant="italic")
        self.src_width:int = int(WIDTH)
        self.src_height: int = int(HEIGHT)
        self.EXPAND_RATIO = 0.7
        self.vis_width = int(self.src_width*self.EXPAND_RATIO)
        self.vis_height = int(self.src_height*self.EXPAND_RATIO)

        self.close_flag = False
        self.close_ok = False
        
        self.ret = True

        # * シェアードメモリ関連
        try:
            self.mm = mmap.mmap(fileno=-1, length=20000, tagname="SAMoSSharedMemory", access=mmap.ACCESS_WRITE)
        except :
            try:
                self.mm = mmap.mmap(fileno=-1, length=20000, tagname="SAMoSSharedMemory", access=mmap.ACCESS_WRITE, create=True) #type:ignore
            except :
                print("シェアードメモリーの作成に失敗しました")
                self.ret=False

        # * ビデオキャプチャの準備
        if with_camera:
            url: str = camera_url
            # RTMPサーバーを経由して動画を取り込む場合のfpsは29.97002997002997
            fps = 29.97002997002997  # TODO 暫定
            self.load_interval: float = 1000/fps  # 画面更新のインターバル[ms]
        else:
            url: str = camera_url
            fps = FPS
            self.load_interval: float = 1000 / fps
        self.url = url
        self.capture = ImageInput(path=url, with_basler=with_camera, batch_size = BATCH_SIZE)

        size = (WIDTH,HEIGHT)
        self.writer = cv2.VideoWriter(dst_movie_path, format, FPS, size)
        
        self.create_widget()
        
        if self.ret == False or self.capture.ret == False:
            # ウィンドウを閉じる
            self.master.destroy()  # type:ignore
            if __name__ == "__main__":
                self.master.quit()
                
        else:
            # * 表示の開始
            FIGURE_RELOAD_SEC = 1.0
            self.flow_reload_interval = fps * FIGURE_RELOAD_SEC
            self.src = None
            self.reload()

    def create_widget(self):
        # 動画用の作成
        self.window_margin = 10
        self.movie_canvas = tk.Canvas(self,
                                width=self.vis_width,
                                height=self.vis_height,
                                background="white")
        self.movie_canvas.grid(row=0, column=0,rowspan=2,sticky=tk.NSEW)

        # master_statusの表示
        self.lbl_master_status = ttk.Label(self,
                                           text="録画中",
                                           background="white",
                                           font=self.font_status,
                                           anchor='center',
                                           relief='groove')
        self.lbl_master_status.grid(
            row=0, column=1,
            padx = 5, pady=10, sticky=tk.NSEW)
        
        self.stop_button = tk.Button(
            self, text="停止", command=self.stop, 
            font = self.font_index,relief=tk.RAISED,bd=8)
        self.stop_button.grid(
            row=1, column=1, 
            padx=5,pady=5,sticky=tk.NSEW)
        
        # Frame自身もトップレベルウィジェットに配置
        self.grid(column=0, row=0, sticky=tk.NSEW)

    def reload(self):
        if not self.close_flag:
            reload_start_time = time.perf_counter()
            
            # captureの更新
            self.capture.load()

            # 画像取得状態の確認
            success_flag = self.capture.suc_flag
            if success_flag:
                # 画像取得と変換
                self.src = self.capture.img
                if self.src is None:
                    print("Error: Frame not captured correctly.")
                    return

                resized_img = cv2.resize(self.src, (self.vis_width, self.vis_height))
                img_pil = utl.arr_to_pil(resized_img, False)
                img_itk = utl.pil_to_itk(img_pil=img_pil, master=self.movie_canvas)

                # 画像表示
                self.movie_canvas.delete("fig")
                self.movie_canvas.photo = img_itk  # type:ignore
                self.movie_canvas.create_image(0, 0, anchor=tk.NW, image=self.movie_canvas.photo, tags='fig')  # type:ignore

                # * 録画用image取得
                dst = cv2.cvtColor(self.src, cv2.COLOR_RGB2BGR)
                self.writer.write(dst)

                # * レシピ変更の検出
                if self.mm[2000] == 1:
                    new_recipe_name = self.mm[2004:2515].decode('sjis', 'ignore').replace('\0', '')
                    print("recipe changed to " + new_recipe_name + "...")
                    self.mm[2000] = 0
                    self.reset(new_recipe_name)
            else:
                print("Frame capture failed, retrying...")
                self.capture.release()
                return

            reload_end_time = time.perf_counter()

            # * 所要時間計算
            self.td_reloading = reload_end_time - reload_start_time

            # * Recall
            self.delay_time = round(self.load_interval - self.td_reloading * 1000)
            if self.delay_time < 0:
                self.delay_time = 0

            self.after(self.delay_time, self.reload)
        else:
            self.close_ok = True


    def reset(self,new_recipe_name="unknown"):
        
        print('reset start')

        self.close_flag = True
        
        self.mm.close()

        # 各種コンストラクタを開放
        self.writer.release()
        self.capture.release()

        print("カメラ接続状態:",self.capture.camera.IsOpen())

        # * 再初期化
        config = utl.Config()
        
        if self.with_camera and new_recipe_name != "":
            cur_line = "04"
            target_recipe_path = "./_model/" + cur_line + "/" + new_recipe_name + '/' + new_recipe_name + '.json'
            print(target_recipe_path)
            config_path = config.load_data(target_recipe_path)
        else:
            cur_line = "04"
            target_recipe_path = "./_model/" + cur_line + "/_/_.json"
            print(target_recipe_path)
            config_path = config.load_data()
        
        if config_path != "":
            print("解析モードで再起動します")
            prog_interface = ProgressInterface(config=config, 
                                               master=self.master,
                                               with_camera=self.with_camera,
                                               analysis_batch=4, 
                                               output_speed_data=True,
                                               reset=False)
        else:
            print("録画モードで再起動します")
            recorder = Recorder(master=self.master,
                                product_name=new_recipe_name,
                                with_camera=self.with_camera,
                                camera_url=self.url)
            
    def stop(self):
        
        print("Recording stop")

        self.close_flag = True
        
        self.mm.close()

        # 各種コンストラクタを開放
        self.writer.release()
        self.capture.release()

        print("カメラ接続状態:",self.capture.camera.IsOpen())


        self.master.destroy()        
        if __name__ == "__main__":
            self.master.quit()

        print('Finalized.\n')

        if __name__ == "__main__":
            exit()



# * for debugger
if __name__ == "__main__":
    print('--- Work Inference---')
    # コンフィグデータ読み取り
    config = utl.Config()
    config_path = config.load_data()

    if config_path != "":

        # tkフレームの作成
        wi_root = tk.Tk()
        # tkフレーム設定
        window_title: str = "Work Inference | SAMoS DEBUG"
        wi_root.title(window_title)
        wi_root.state("zoomed")
        prog_interface = ProgressInterface(
            config=config,  master=wi_root,rec_tgt=window_title,
            with_camera=False,analysis_batch=4, output_speed_data=True)
        wi_root.protocol('WM_DELETE_WINDOW', prog_interface.close)

        # mainloopの開始
        wi_root.mainloop()

    else:
        # tkフレームの作成
        wi_root = tk.Tk()
        # tkフレーム設定
        window_title: str = "Work Inference | SAMoS DEBUG"
        wi_root.title(window_title)
        wi_root.state("zoomed")
        recorder = Recorder(
            master=wi_root,
            with_camera=False)

        wi_root.protocol('WM_DELETE_WINDOW', recorder.stop)


        # mainloopの開始
        wi_root.mainloop()

