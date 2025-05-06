import os
import pickle
import shutil
import gc
import json
import re
from collections import OrderedDict
from tabnanny import check
from typing import List
from pathlib import Path

# tensorflowのInformationを非表示に
# tfのimport前に入れること。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
import tkinter.messagebox as messagebox
from imageio import imread
from matplotlib import cm as cm
from matplotlib import pyplot as plt
import japanize_matplotlib
from natsort import natsorted
from PIL import Image
from keras import backend as K
from keras.models import load_model
from keras.optimizers import adam_v2
from keras.utils import load_img, img_to_array

if __name__ == "__main__":
    from samos_utils import Config

    from SSD_pierluigiferrari.models.keras_ssd300 import ssd_300
    from SSD_pierluigiferrari.keras_loss_function.keras_ssd_loss import SSDLoss
    from SSD_pierluigiferrari.keras_layers.keras_layer_AnchorBoxes import \
        AnchorBoxes
    from SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetections import \
        DecodeDetections
    from SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetectionsFast import \
        DecodeDetectionsFast
    from SSD_pierluigiferrari.keras_layers.keras_layer_L2Normalization import \
        L2Normalization

    from SSD_pierluigiferrari.ssd_encoder_decoder.ssd_output_decoder import \
        decode_detections, decode_detections_fast

    from SSD_pierluigiferrari.data_generator.object_detection_2d_data_generator import \
        DataGenerator
    from SSD_pierluigiferrari.data_generator.object_detection_2d_photometric_ops import \
        ConvertTo3Channels
    from SSD_pierluigiferrari.data_generator.object_detection_2d_patch_sampling_ops import \
        RandomMaxCropFixedAR
    from SSD_pierluigiferrari.data_generator.object_detection_2d_geometric_ops import \
        Resize
    from SSD_pierluigiferrari.data_generator.object_detection_2d_misc_utils import \
        apply_inverse_transforms

else:
    from .samos_utils import Config

    from .SSD_pierluigiferrari.models.keras_ssd300 import ssd_300
    from .SSD_pierluigiferrari.keras_loss_function.keras_ssd_loss import SSDLoss
    from .SSD_pierluigiferrari.keras_layers.keras_layer_AnchorBoxes import \
        AnchorBoxes
    from .SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetections import \
        DecodeDetections
    from .SSD_pierluigiferrari.keras_layers.keras_layer_DecodeDetectionsFast import \
        DecodeDetectionsFast
    from .SSD_pierluigiferrari.keras_layers.keras_layer_L2Normalization import \
        L2Normalization

    from .SSD_pierluigiferrari.ssd_encoder_decoder.ssd_output_decoder import \
        decode_detections, decode_detections_fast

    from .SSD_pierluigiferrari.data_generator.object_detection_2d_data_generator import \
        DataGenerator
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_photometric_ops import \
        ConvertTo3Channels
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_patch_sampling_ops import \
        RandomMaxCropFixedAR
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_geometric_ops import \
        Resize
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_misc_utils import \
        apply_inverse_transforms


def elem_crop(config: Config, train_ratio: float = 0.8, batch_size = 64, check_mode=False, continuous_mode = False):
    """SSDによる作業学習用画像の切り抜き

    Parameters
    ----------
    config : Config
        設定ファイル
    train_ratio : float, optional
        TrainとValidationの比率\n
        Train枚数/全体枚数で指定, by default 0.8
    check_mode : bool, optional
        切り取り結果をpltにて一枚ずつ確認（デバッグ用）, by default False
    """

    # * 初期設定
    
    # tfのコンフィグ・セッション初期化
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(
                device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")

    # Set the image size.
    img_height: int = 300
    img_width: int = 300
    img_channels: int = 3

    # configデータ読み取り
    element_names:list[str] = []
    max_nums = []
    max_widths = []
    max_heights = []
    element_names, max_widths, max_heights, max_nums = config.get_element_temp_info()
    n_classes: int = config.data["elem_num"]

    # 学習のための振り分け設定
    validation_step = int(1/(1-train_ratio))

    # src画像フォルダの指定,読込リストの作成
    src_img_dir = Path(config.data["work_annotation_src_img_dir"])
    src_img_paths: list[Path] = natsorted(
        list(src_img_dir.glob('*.jpg')), key=lambda x: x.name)
    batched_img_paths = []

    q, mod = divmod(len(src_img_paths), batch_size)
    if not mod == 0:
        q += 1
    for n in range(q):
        temp_paths = src_img_paths[n*batch_size:(n+1)*batch_size-1]
        batched_img_paths.append(temp_paths)
    
    print('number of batches: ', q)

    # sortの準備
    columns = ['label', 'conf', 'xmin', 'ymin', 'xmax', 'ymax']

    # work_annotation_result.csv読み込み
    csv_path: str = config.data["root_dir"] + "/work/labels.csv"
    dst_df: pd.DataFrame = pd.read_csv(
        csv_path, index_col=0, encoding="utf_8_sig")
    print(dst_df)

    #　dst(each elements)画像の格納先の作成・取得
    work_train_src_root = Path(config.data["work_train_src_img_dir"])
    
    # clear work_train_src
    shutil.rmtree(work_train_src_root)
    os.makedirs(work_train_src_root, exist_ok=True)
    
    # make sub dir
    for e_name in element_names:
        train_root = work_train_src_root / e_name / 'train'
        val_root = work_train_src_root / e_name / 'validation'
        unique_w = dst_df[e_name].unique()
        unique_str_w = unique_w.astype(str)
        w_list: List[str] = unique_str_w.tolist()
        for i,w in enumerate(w_list):
            if "|" in w:
                classes = re.split(r"[|]",w)
                for c in classes:
                    w_list.append(c)
        w_list = [i for i in w_list if not "|" in i]
        set(w_list)
        for w in w_list:
            os.makedirs(name=(train_root / str(w)), exist_ok=True)
            os.makedirs(name=(val_root / str(w)), exist_ok=True)

    # 統計データ準備
    stat_train = np.zeros((int(config.data['elem_num']),
                           int(config.data['work_num'])), dtype=int)
    stat_test = np.zeros((int(config.data['elem_num']),
                          int(config.data['work_num'])), dtype=int)
    
    # check用変数
    count = 0

    # * 1. Load a trained SSD

    # * 1.1. Build the model and load trained weights into it

    # 1: Build the Keras model

    model: tf.keras.models.Model = ssd_300(image_size=(img_height, img_width, img_channels),  # type:ignore
                                           n_classes=n_classes,
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
                                           steps=[8, 16, 32, 64, 100, 300],
                                           offsets=[0.5, 0.5, 0.5,
                                                    0.5, 0.5, 0.5],
                                           clip_boxes=False,
                                           variances=[0.1, 0.1, 0.2, 0.2],
                                           normalize_coords=True,
                                           subtract_mean=[123, 117, 104],
                                           swap_channels=[2, 1, 0],
                                           confidence_thresh=0.5,
                                           iou_threshold=0.45,
                                           top_k=200,
                                           nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    # TODO: Set the path of the trained weights.
    weights_path: str = config.data["elem_weight"]

    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss) #type:ignore

    # * 以下ミニバッチ処理

    for n in range(q):

        print("progress: {}/{}:".format(n+1,q))

        # * 2. Load some images

        orig_images = []  # Store the images here.
        input_images = []  # Store resized versions of the images here.

        for img_path in batched_img_paths[n]:
            orig_images.append(imread(img_path))
            img = load_img(
                img_path, target_size=(img_height, img_width))
            img = img_to_array(img)
            input_images.append(img)

        input_images = np.array(input_images)

        # * 3. Make predictions

        print("now predicting...")

        y_pred: np.ndarray = model.predict(input_images) # type:ignore

        confidence_threshold = 0.6

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                         for k in range(y_pred.shape[0])]
        
        # debug
        # print(y_pred_thresh)

        # * 4. Visualize the predictions
        if check_mode:
        # if check_mode and count == 0:

            count += 1

            np.set_printoptions(precision=2, suppress=True,  # type:ignore
                                linewidth=90)  # type:ignore

            # Display the image and draw the predicted boxes onto it.

            # Set the colors for the bounding boxes
            # 描画設定(checkmode用)
            colors = cm.hsv(np.linspace(  # type:ignore
                0, 1, n_classes+1, endpoint=False)).tolist()

            plt.figure(figsize=(20, 12))
            plt.imshow(orig_images[0])

            current_axis = plt.gca()

            for box in y_pred_thresh[0]:
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = box[2] * orig_images[0].shape[1] / img_width
                ymin = box[3] * orig_images[0].shape[0] / img_height
                xmax = box[4] * orig_images[0].shape[1] / img_width
                ymax = box[5] * orig_images[0].shape[0] / img_height
                color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(
                    element_names[int(box[0]-1)], box[1])
                current_axis.add_patch(plt.Rectangle(  # type:ignore
                    (xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
                current_axis.text(xmin, ymin, label, size='x-large',
                                  color='white', bbox={'facecolor': color, 'alpha': 1.0})

            plt.show()

        
        # * 切り取り操作！
        
        # 処理中の通知
        print('now cropping...\n')
        
        # * バッチ内全画像に対する処理
        for i, preds in enumerate(y_pred_thresh):

            row = n*batch_size + i

            if i % validation_step == 0:
                str_trainval = "validation"
            else:
                str_trainval = "train"

            # 画像のdataをDataFrameで取得、confでsort
            # ! label[0] は background. elements[1] = label[0]
            df = pd.DataFrame(data=preds, columns=columns)
            
            img_results = df.sort_values('conf',ascending=False).values
            det_label = img_results[:, 0]
            det_conf = img_results[:, 1]
            det_xmin = img_results[:, 2]
            det_ymin = img_results[:, 3]
            det_xmax = img_results[:, 4]
            det_ymax = img_results[:, 5]

            # 各要素のカウント変数を初期化
            each_element_counts = [0] * config.data['elem_num']

            # 各抽出要素に対する処理
            for j in range(det_conf.shape[0]):
                
                xmin = det_xmin[j] * orig_images[i].shape[1] / img_width
                ymin = det_ymin[j] * orig_images[i].shape[0] / img_height
                xmax = det_xmax[j] * orig_images[i].shape[1] / img_width
                ymax = det_ymax[j] * orig_images[i].shape[0] / img_height

                label = int(det_label[j])
                each_element_counts[label-1] += 1

                # 各要素最大個数までの処理
                if each_element_counts[label-1] <= max_nums[label-1]:

                    # 切り取りの実施
                    # （幅はmax_widthとmax_height）で決まる
                    center = [(ymin + ymax) / 2, (xmin + xmax) / 2]
                    x1 = int(center[1] - max_widths[label-1] / 2)
                    x2 = int(center[1] + max_widths[label-1] / 2)
                    y1 = int(center[0] - max_heights[label-1] / 2)
                    y2 = int(center[0] + max_heights[label-1] / 2)
                    # intへの丸めで x2 - x1 != width, y2 - y1 != heightとなった場合の調整
                    temp_width = x2 - x1
                    temp_height = y2 - y1
                    if temp_width != max_widths[label-1]:
                        x2 = x2 + (max_widths[label-1] - temp_width)
                    if temp_height > max_heights[label-1]:
                        y2 = y2 + (max_heights[label-1] - temp_height)
                    save_image = Image.fromarray(orig_images[i]).crop((x1, y1, x2, y2))
                    # print(x1,y1,x2,y2)

                    # 保存先決定
                    designated_dst = str(dst_df.iat[row,label-1])
                    # print(designated_dst)
                    designated_classes = re.split(r'[|]',designated_dst)
                    for c in designated_classes:
                        # print(c)
                        dst_file_name = element_names[label-1] + \
                            "_" + dst_df.index[row]
                        dst_path = work_train_src_root / \
                            element_names[label-1] / str_trainval / \
                            c / dst_file_name #type:ignore

                        save_image.save(dst_path, quality=95)

                    # # 統計データへの加算
                    # if str_trainval == 'train':
                    #     stat_train[label-1, dst_df.iloc[row, label-1]] += 1
                    # else:
                    #     stat_test[label-1, dst_df.iloc[row, label-1]] += 1

        

    # * 結果表示
    if check_mode:
        print('crop is finished.\n')
        print('---Result---')
        df_train = pd.DataFrame(stat_train.T)
        df_train.columns = element_names
        df_train.index = config.data['work_name']
        df_test = pd.DataFrame(stat_test.T)
        df_test.columns = element_names
        df_test.index = config.data['work_name']
        print('---Train---')
        print(df_train)
        print('---test---')
        print(df_test)

    # * 各検出要素の分類クラスをjsonに追記
    elem_class_dict = OrderedDict()
    work_train_src_root = Path(config.data["work_train_src_img_dir"])
    for name in element_names:
        # 分類クラス定義
        classes: List[str] = list()
        target_dir: Path = work_train_src_root / name / 'train'
        for p in target_dir.iterdir():
            if p.is_dir():
                classes.append(p.name)
        
        elem_class_dict[name] = classes
        
        for e in config.data["element"]:
            if e["name"] == name:
                e["classes"] = elem_class_dict[name]
    
    # 設定ファイル更新
    json_str = json.dumps(config.data, ensure_ascii=False, indent=4)
    with open(config.data["config_default_path"], 'w', encoding='utf-8') as f:
        f.write(json_str)

    # * メモリ解放
    tf.keras.backend.clear_session()
    del model
    gc.collect()

    if not continuous_mode:
        messagebox.showinfo("切り取り完了", "検出対象の切り取りが完了しました")

    # # * log出力
    # dst_log = Path(config.data["elem_crop_log"])
    # df_train.to_csv(dst_log, encoding='utf_8_sig', mode='w')
    # df_test.to_csv(dst_log, encoding='utf_8_sig', mode='a')

# * for debugger
if __name__ == "__main__":
    print('--- Element Cropper ---')
    config = Config()
    config.load_data()
    elem_crop(config=config, check_mode=False,continuous_mode=False)
