import os
import datetime
import json
import gc
from math import ceil
from pathlib import Path
from tkinter import messagebox
from typing import List
import glob
from tqdm import tqdm

import cv2
import random
import xml.dom.minidom as md
import xml.etree.ElementTree as et
import pathlib
import shutil

# tensorflowのInformationを非表示に
# tfのimport前に入れること。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

if __name__ == "__main__":
    import samos_utils as utl
    from SSD_pierluigiferrari.data_generator.data_augmentation_chain_original_ssd import \
        SSDDataAugmentation
    from SSD_pierluigiferrari.data_generator.object_detection_2d_data_generator import \
        DataGenerator
    from SSD_pierluigiferrari.data_generator.object_detection_2d_geometric_ops import \
        Resize
    from SSD_pierluigiferrari.data_generator.object_detection_2d_misc_utils import \
        apply_inverse_transforms
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
    from SSD_pierluigiferrari.ssd_encoder_decoder.ssd_input_encoder import \
        SSDInputEncoder
    from SSD_pierluigiferrari.ssd_encoder_decoder.ssd_output_decoder import (
        decode_detections, decode_detections_fast)

else:
    from . import samos_utils as utl
    from .SSD_pierluigiferrari.data_generator.data_augmentation_chain_original_ssd import \
        SSDDataAugmentation
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_data_generator import \
        DataGenerator
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_geometric_ops import \
        Resize
    from .SSD_pierluigiferrari.data_generator.object_detection_2d_misc_utils import \
        apply_inverse_transforms
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
    from .SSD_pierluigiferrari.ssd_encoder_decoder.ssd_input_encoder import \
        SSDInputEncoder
    from .SSD_pierluigiferrari.ssd_encoder_decoder.ssd_output_decoder import (
        decode_detections, decode_detections_fast)

class Element:
    """アノテーション対象要素クラス

    [extended_summary]

    Attribute:

    """

    def __init__(self, name: str, xmin=0, ymin=0, xmax=0, ymax=0, is_fixed=False):
        """アノテーション対象要素クラスの初期化

        Args:
            name (str): 対象の名前
            xmin (int, optional): 矩形を構成する点P1のx座標. Defaults to 0.
            ymin (int, optional): 矩形を構成する点P1のy座標. Defaults to 0.
            xmax (int, optional): 矩形を構成する点P2のx座標. Defaults to 0.
            ymax (int, optional): 矩形を構成する点P2のy座標. Defaults to 0.
            is_fixed (bool, optional): 次フレームに残すかどうかのフラグ. Defaults to False.
        """

        self.name = name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.is_fixed = is_fixed

class UpdateConfig(tf.keras.callbacks.Callback):
    """
    コンフィグ.json更新用コールバッククラス

    各epochの最後に、更新があればjsonを更新し出力します
    """    
    
    def __init__(self, config: utl.Config, dst_weight_path_str: str):
        self.config = config
        self.dst_weight_path_str = dst_weight_path_str
        self.val_loss = []
        self.best_val_loss = None

    def on_epoch_end(self, epoch, logs={}):
        
        self.val_loss.append(logs['val_loss'])
        self.best_val_loss = min(self.val_loss)

        if not (len(self.val_loss) > 1 and self.val_loss[-1] != self.best_val_loss):

            # jsonを修正し出力する
            self.config.data["elem_trained_epoch"] = epoch+1
            self.config.data["elem_weight"] = self.dst_weight_path_str
            json_str = json.dumps(
                self.config.data, ensure_ascii=False, indent=4)
            with open(self.config.data["config_default_path"], 'w', encoding='utf-8') as f:
                f.write(json_str)
            
class ClearMemory(tf.keras.callbacks.Callback):
    """
    メモリ解放用コールバッククラス

    各epochの最後に明示的にガベージコレクション、クリアセッション
    """
    def on_epoch_end(self, epoch,logs=None):
        gc.collect()
        keras.backend.clear_session()


def make_train_val_list(config: utl.Config, train_ratio: float = 0.8, dst_dir: Path = None): #type:ignore
    """
    dataset全体のファイル名を記述したdatasets.txtから、
    train用のtrain_datasets.txtとvalidation用のval_datasets.txtを作成します

    全体を100%としてtrain_ratioの比率でランダムでピックアップし、
    datasets.txtから抽出したものをval_datasets.txtとして出力、
    抽出されなかった残りの部分をtrain_datasets.txtとして出力します。

    抽出の乱数シードは固定しています。

    デフォルトでは、datasets.txtと同じフォルダに出力されます。
    任意の場所に変更したい場合は、出力先ディレクトリを引数dst_dirに入力してください。

    Parameters
    ----------
    config : utl.Config
        作成するデータセットのコンフィグファイル
    train_ratio : float, optional
        全体を100%としたとき、validation用に抽出する割合, by default 0.8
    dst_dir : Path, optional
        出力先ディレクトリ、指定の無い場合はdatasets.txtと同じディレクトリに出力します, by default None

    Returns
    -------
    bool
        作成が成功したかどうかのT/F
    """

    # src読込
    src_path = Path(config.data["src_imageset"])
    df_src: pd.DataFrame = pd.read_csv(src_path, header=None)  # type:ignore

    # 出力先定義
    if dst_dir == None:
        dst_train_path = Path(config.data["train_imageset"])
        dst_val_path = Path(config.data["val_imageset"])
    else:
        if dst_dir.is_dir():
            dst_train_path = Path(dst_dir / "train_dataset.txt")
            dst_val_path = Path(dst_dir / "val_dataset.txt")
        else:
            messagebox.showerror("指定された保存先ディレクトリは無効です")
            return False

    # ランダム抽出でtrain_imagesetを作成
    df_train = df_src.sample(frac=train_ratio, random_state=0)
    # ソースとランダム抽出の和集合の差を取ってval_imagesetを作成
    df_val = df_src.merge(right=df_train, indicator=True, how="outer").query(
        '_merge=="left_only"').drop(labels='_merge', axis=1)

    # ソート
    df_train = df_train.sort_index()
    df_val = df_val.sort_index()

    # 出力
    print(df_train)
    print(df_val)
    df_train.to_csv(dst_train_path, header=False, index=False)
    df_val.to_csv(dst_val_path, header=False, index=False)
    print(str(dst_train_path)+"が出力されました")
    print(str(dst_val_path)+"が出力されました")

    return True

class MakeAugmentedImg:

    def __init__(self,config:utl.Config):

        annotations_dir_path = config.data['elem_annotation_results_dir']
        self.imageset_path = config.data['src_imageset']
        elem_train_src_dir_path = config.data['elem_train_src_img_dir']

        self.annotations_list = glob.glob(annotations_dir_path+'/*.xml')
        self.src_image_list = glob.glob(elem_train_src_dir_path+'/*.jpg')

        self.imagesets_path = config.data['src_imageset']
        self.elem_train_src_dir = pathlib.Path(config.data["elem_train_src_img_dir"])
        self.elem_annotation_results_dir = config.data["elem_annotation_results_dir"]

        self.src_width = config.data['src']['width']
        self.src_height = config.data['src']['height']
        self.src_depth = config.data['src']['depth']

    def make_augment_img(self):

        for a in tqdm(self.annotations_list):

            with open(a, encoding="shift_jis") as file:
                xml = file.read()
            
            root = et.fromstring(xml)
            
            folder_name = str(root.find('folder').text)
            file_name = str(root.find('filename').text)

            try:
                augmented_flag = bool(str(root.find('augmented').text))
                if augmented_flag:
                    continue
            except:
                pass

            src_img_path = str(self.elem_train_src_dir) +'\\'+ str(file_name)
            src_img = cv2.imread(src_img_path)

            for i, obj in enumerate(root.findall('object')):

                elem_name = str(obj.find('name').text)

                for bndbox in obj.findall('bndbox'):
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                # 出力の設定
                src_path = pathlib.Path(file_name)
                basename: str = src_path.stem
                aug_name: str = basename + '_' + str(i) + '.jpg'
                dst_aug_path: pathlib.Path = self.elem_train_src_dir / aug_name

                src_elem_list = []
                element = Element(elem_name,xmin,ymin,xmax,ymax)
                src_elem_list.append(element)

                # ランダムに配置しなおした画像の作成
                dst_elem_list = self.randomize(src_img,element)
                dst_img = self.collage(src_img, src_elem_list,dst_elem_list)

                # augmented_img
                cv2.imwrite(str(dst_aug_path), dst_img)

                # 現在の情報の保存
                src_img_basename = os.path.basename(str(dst_aug_path))
                self.add_imagesets(self.imagesets_path, src_img_basename)
                self.make_xml(src_img_basename, dst_elem_list,folder_name)

    def randomize(self,src_img, element:Element):
        """各要素の矩形をランダムに配置可能な座標を得る

        各要素の矩形をランダムに配置可能な座標を得る。
        得られた座標はself.elem_data_listの形を保ったままself.rand_data_listに。

        Returns:
            list: 名前・矩形の左上と右下の座標を含んだ要素のlist
        """

        dst_elem_list: List[Element] = list()

        tmp_arr = np.zeros((480, 640))
        x_rand = random.randint(0, self.src_width - element.width)
        y_rand = random.randint(0, self.src_height - element.height)
        temp_w = abs(element.xmax - element.xmin)
        temp_h = abs(element.ymax - element.ymin)

        dst_elem_list.append(
            Element(element.name,
                    x_rand, y_rand, x_rand+temp_w, y_rand+temp_h))

        return dst_elem_list

    def collage(self, src_img:np.ndarray, src_elem_list, dst_elem_list):
        """elem_listに従いimg_cvから切り抜き、rand_elem_listに従いグレーバックに貼り付ける

        Parameters
        ----------
        dst_elem_list : List
            ランダムな配置にするための座標が記述されたリスト

        Returns
        -------
        ndarray
            要素がグレーバックにランダムに配置された画像
        """
        # * 背景画像の生成（gray_back）
        dst_img = np.zeros((self.src_height, self.src_width, 3))
        dst_img[::] = [122, 122, 122]

        for src, dst in zip(src_elem_list, dst_elem_list):
            trim_img = src_img[src.ymin:src.ymax, src.xmin:src.xmax]
            dst_img[dst.ymin:dst.ymax, dst.xmin:dst.xmax] = trim_img

        return dst_img
    
    def add_imagesets(self, path, basename):
        """imageset.txtにJPEGImageの対象画像を書き込む

        Parameters
        ----------
        dstdir : str
            imageset.txtを含む出力先のディレクトリパス
        basename : str
            画像ファイルのbasename
        """
        with open(path, mode='a') as f:
            f.write(str(os.path.splitext(basename)[0]) + '\n')

    def make_xml(self, basename, elem_list, folder_name):
        """VOC07準拠のxmlファイルを作成する

        出力フォルダは__init__のself.xml_dirを利用

        Parameters
        ----------
        basename : str
            読み込んだ画像ファイルのbasename (e.g. foo.jpg)
        elem_list : List
            xmlに出力するelement群のlist。
        """
        # xmlパスの設定
        xml_file = str(self.elem_annotation_results_dir) + '/' + \
            str(os.path.splitext(basename)[0]) + '.xml'

        # xml構成の作成（~segmentedまで）
        root = et.Element('Annotation')
        folder = et.SubElement(root, 'folder')
        folder.text = folder_name
        filename = et.SubElement(root, 'filename')
        filename.text = str(basename)
        size = et.SubElement(root, 'size')
        width = et.SubElement(size, 'width')
        height = et.SubElement(size, 'height')
        depth = et.SubElement(size, 'depth')
        width.text = str(self.src_width)
        height.text = str(self.src_height)
        depth.text = str(self.src_depth)  # TODO depthをconfigから取得すること
        segmented = et.SubElement(root, 'segmented')
        segmented.text = '0'
        augmented = et.SubElement(root,'augmented')
        augmented.text = 'True'

        # 要素関係のobjectサブエレメントの構成
        for e in elem_list:
            obj = et.SubElement(root, 'object')
            name = et.SubElement(obj, 'name')
            name.text = e.name
            pose = et.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = et.SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = et.SubElement(obj, 'difficult')
            difficult.text = '1'

            bbox = et.SubElement(obj, 'bndbox')
            xmin = et.SubElement(bbox, 'xmin')
            ymin = et.SubElement(bbox, 'ymin')
            xmax = et.SubElement(bbox, 'xmax')
            ymax = et.SubElement(bbox, 'ymax')

            xmin.text = str(e.xmin)
            ymin.text = str(e.ymin)
            xmax.text = str(e.xmax)
            ymax.text = str(e.ymax)

        # xmlの作成（改行するため出力にminidomを利用）
        with open(xml_file, 'w') as f:
            f.write(self.prettify(root))

    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element.

        Parameters
        ----------
        elem : element
            ElementTreeのroot element
        """
        rough_string = et.tostring(elem, 'utf-8')
        reparsed = md.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

def elem_training(config: utl.Config, epoch_num=30, nb_batch=2, patience=5, light_mode: bool = True, continuous_mode:bool=False):
    """検出対象要素の学習を行います。

    elem_annotatorにてアノテーションされた情報（.xml）,
    対象の画像のセットを指し示すimagesets,画像の実体の３種の情報から、
    物体検出のための学習を行います。

    Parameters
    ----------
    config : utl.Config
        該当品番の設定ファイル.jsonを読み込んだConfigクラス
    epoch_num : int, optional
        学習を行うepoch数, by default 30
    nb_batch : int, optional
        学習の際のバッチサイズ, by default 2
    patience : int, optional
        学習の際、val_lossに数epoch改善が見られない場合に
        EarlyStoppingを実行するepoch数, by default 5
    light_mode : bool, optional
        早期終了ならびに、出力ファイル数を減らすかどうかのモードの選択を行います, by default True
            True: 
                save_best_only = True in model.fit
                add EarlyStopping to model.callbacks (patience = 3)
            False: 
                save_best_only = False in model.fit

    """

    # config読み取り
    conf = config

    # random画像の作成
    # print('学習用の拡張画像を作成します')
    # mai = MakeAugmentedImg(conf)
    # mai.make_augment_img()
    # print('学習用の拡張画像の作成が完了しました')
    
    # # datasetの確認
    train_imageset_path = Path(conf.data["train_imageset"])
    val_imageset_path = Path(conf.data["val_imageset"])
    try:
        make_train_val_list(conf)
    except:
        messagebox.showerror("","[train/val]_imageset.txtを作成することができませんでした。\nプログラムを終了します。")
        return None

    # * tfのコンフィグ・セッション初期設定
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(
                device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")

    # * 1. Set the model configuration parameters

    img_height = 300  # Height of the model input images
    img_width = 300  # Width of the model input images
    img_channels = 3  # Number of color channels of the model input images
    # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    mean_color = [123, 117, 104]
    # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    swap_channels = [2, 1, 0]
    # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
    n_classes = conf.data["elem_num"] # type:ignore
    # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
    scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
    scales = scales_pascal
    aspect_ratios = [[1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],
                     [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
    two_boxes_for_ar1 = True
    # The space between two adjacent anchor box center points for each predictor layer.
    steps = [8, 16, 32, 64, 100, 300]
    # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    clip_boxes = False
    # The variances by which the encoded target coordinates are divided as in the original implementation
    variances = [0.1, 0.1, 0.2, 0.2]
    normalize_coords = True

    # * 2. Build or load the model

    # 1: Build the Keras model.
    keras.backend.clear_session()  # Clear previous models from memory.
    model: tf.keras.models.Model = ssd_300(image_size=(img_height, img_width, img_channels),  # type:ignore
                                           n_classes=n_classes,
                                           mode='training',
                                           l2_regularization=0.0005,
                                           scales=scales,
                                           aspect_ratios_per_layer=aspect_ratios,
                                           two_boxes_for_ar1=two_boxes_for_ar1,
                                           steps=steps,
                                           offsets=offsets,
                                           clip_boxes=clip_boxes,
                                           variances=variances,
                                           normalize_coords=normalize_coords,
                                           subtract_mean=mean_color,
                                           swap_channels=swap_channels)

    # 2: Load some weights into the model.
    # TODO: Set the path to the weights you want to load.

    # 初回学習時にはsampled_weightを、学習再開時にはelem_weightをロードする
    if conf.data["elem_trained_epoch"] > 0 and conf.data["elem_weight"] != "":
        weight_path = conf.data["elem_weight"]
    else:
        weight_path = conf.data["sampled_weight"]
    model.load_weights(weight_path, by_name=True)

    # 3: Instantiate an optimizer and the SSD loss function and compile the model.
    #    If you want to follow the original Caffe implementation, use the preset SGD
    #    optimizer, otherwise I'd recommend the commented-out Adam optimizer.
    adam = keras.optimizers.adam_v2.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = keras.optimizers.SGD(
        lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss) #type:ignore

    # * 3. Set up the data generators for the training
    # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
    # Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.
    train_dataset = DataGenerator(
        load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(
        load_images_into_memory=False, hdf5_dataset_path=None)

    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.
    # TODO: Set the paths to the datasets here.

    # The directories that contain the images.
    images_dir = conf.data["elem_train_src_img_dir"]

    # The directories that contain the annotations.
    annotations_dir = conf.data["elem_annotation_results_dir"]

    # The paths to the image sets.
    train_image_set_filename = str(train_imageset_path)
    val_image_set_filename = str(val_imageset_path)

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    elements = conf.data["element"]
    classes: List[str] = list()
    classes.append('background')
    for e in elements:
        classes.append(e["name"])

    train_dataset.parse_xml(images_dirs=[images_dir],
                            image_set_filenames=[train_image_set_filename],
                            annotations_dirs=[annotations_dir],
                            classes=classes,
                            include_classes='all',
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)
    val_dataset.parse_xml(images_dirs=[images_dir],
                          image_set_filenames=[val_image_set_filename],
                          annotations_dirs=[annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)

    # * Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
    # * speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
    # * option in the constructor, because in that cas the images are in memory already anyway. If you don't
    # * want to create HDF5 datasets, comment out the subsequent two function calls.
    # train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_trainval.h5',
    #                                 resize=False,
    #                                 variable_image_size=True,
    #                                 verbose=True)
    # val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
    #                                 resize=False,
    #                                 variable_image_size=True,
    #                                 verbose=True)

    # 3: Set the batch size.
    # Change the batch size if you like, or if you run into GPU memory issues.
    batch_size = nb_batch

    # 4: Set the image transformations for pre-processing and data augmentation options.

    # For the training generator:
    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)

    # For the validation generator:
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)

    # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                       model.get_layer('fc7_mbox_conf').output_shape[1:3],
                       model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                       model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

    ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        steps=steps,
                                        offsets=offsets,
                                        clip_boxes=clip_boxes,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=normalize_coords)

    # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             transformations=[
                                                 ssd_data_augmentation],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

    # Get the number of samples in the training and validations datasets.
    train_dataset_size = train_dataset.get_dataset_size()
    val_dataset_size = val_dataset.get_dataset_size()

    print("Number of images in the training dataset:\t{:>6}".format(
        train_dataset_size))
    print("Number of images in the validation dataset:\t{:>6}".format(
        val_dataset_size))

    # * 4. Set the remaining training parameters

    dst_element_dir = Path(conf.data["elem_weights_dir"])
    dst_weight_path_str = str(dst_element_dir) + "/" + conf.data["title"] + "_elements.hdf5"

    # Define model callbacks.
    # TODO: Set the filepath under which you want to save the model.
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=dst_weight_path_str,
                                                       monitor='val_loss',
                                                       verbose=1,
                                                       save_best_only=light_mode,
                                                       save_weights_only=True,
                                                       mode='auto',
                                                       save_freq='epoch')

    history_filepath = Path(conf.data["elem_train_log"])
    csv_logger = keras.callbacks.CSVLogger(filename=history_filepath,
                                           separator=',',
                                           append=True)

    # Define a learning rate schedule.
    def lr_schedule(epoch):
        if epoch < 80:
            return 0.001
        elif epoch < 100:
            return 0.0001
        else:
            return 0.00001

    learning_rate_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule,
                                                                    verbose=1)

    terminate_on_nan = keras.callbacks.TerminateOnNaN()

    clear_memory = ClearMemory()

    update_config = UpdateConfig(config=conf,
                                 dst_weight_path_str=dst_weight_path_str)

    if light_mode:

        early_stopping = keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True)

        callbacks = [model_checkpoint,
                     csv_logger,
                     learning_rate_scheduler,
                     terminate_on_nan,
                     early_stopping,
                     clear_memory,
                     update_config]

    else:

        callbacks = [model_checkpoint,
                     csv_logger,
                     learning_rate_scheduler,
                     terminate_on_nan,
                     clear_memory,
                     update_config]

    # * 5. Train
    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
    initial_epoch = conf.data["elem_trained_epoch"]
    final_epoch = initial_epoch + epoch_num

    # * finetuning用に固定
    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
              'conv4_1', 'conv4_2', 'conv4_3', 'pool4']
    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    # * 学習の実行
    history: tf.keras.callbacks.History = model.fit(train_generator, # type:ignore
                                                    steps_per_epoch=ceil(
                                                        train_dataset_size/batch_size),
                                                    epochs=final_epoch,
                                                    callbacks=callbacks,
                                                    validation_data=val_generator,
                                                    validation_steps=ceil(
                                                        val_dataset_size/batch_size),
                                                    initial_epoch=initial_epoch,
                                                    verbose=1)

    keras.backend.clear_session()
    del model
    gc.collect()
    
    if not continuous_mode:
        messagebox.showinfo("学習完了", "要素の学習が完了しました")

    if not light_mode:
        results = pd.DataFrame(history.history)
        results.plot()

    return history


# * for debugger
if __name__ == "__main__":
    print('--- Element Training ---')
    config = utl.Config()
    config.load_data()
    elem_training(config, epoch_num=30, nb_batch=4, patience=50, light_mode=True)
