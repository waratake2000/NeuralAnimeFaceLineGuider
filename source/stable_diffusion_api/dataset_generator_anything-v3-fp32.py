from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import io
import webuiapi

import csv
import  subprocess

import datetime
import pytz

import time
import random


def get_latest_tagname():
    result = subprocess.run(["git","tag","-l"],capture_output=True, text=True).stdout
    result = str(result).split("\n")
    type(result)
    latest_tag = result[-2]
    return latest_tag


def get_current_now_datetime():
    # JSTのタイムゾーンを取得
    jst = pytz.timezone('Asia/Tokyo')
    # 現在の日時をJSTで取得
    now = datetime.datetime.now(jst)
    # 日時を指定された形式にフォーマット
    formatted_date = now.strftime('%m%d_%H%M%S')
    # print(formatted_date)
    return formatted_date

# 画像を読み込む下準備-------------------------------------------------------------------------------

# WFLWのアノテーションデータの読み込み
annotation_data_path = "/root/dataset/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
WFLW_images = "/root/dataset/WFLW/WFLW_images"

# 生成したデータの保存先 画像&csv
image_save_path = "/root/dataset/SD_generated_dataset/anything-v3-fp32/images"
annotation_save_path = "/root/dataset/SD_generated_dataset/anything-v3-fp32/annotations"

# CSVの読み込みとリスト化
anotation_data_lists = [] # 画像データと座標データがまとめられたリスト
with open(annotation_data_path) as f:
    anotation_data = f.readlines()

random.shuffle(anotation_data)

for i in anotation_data:
    anotation_data_lists.append(i.split(" "))

print("len(anotation_data_lists",len(anotation_data_lists))
time.sleep(1)

# stable diffsuion web APIの用意
api = webuiapi.WebUIApi(host='192.168.0.50', port=7860)
# anotation_data_lists


# パラメータ類----------------------------------------------------------------------------------------
load_image_num = range(5000) # csvの列ナンバー
# 画像のクロップマージンの画像サイズに対する割合
mergin_per = 0.2
input_control_image_size = (512,512)
# プロンプトの用意&生成画像サイズ
prompts = [
    "Anime,anime,face",
    "short hair,quality, eyes, best, winning, detailed, face,  Masterpiece, Award, Quality, solo, quality,1girl, masterpiece, anime,Anime,(perfect anime illustration),kawaii,Kawaii,anime,Anime",
    "quality,Boy, eyes, best, winning, detailed, face,  Masterpiece, Award, Quality, solo, quality,1boy, masterpiece, anime,Anime,(perfect anime illustration),boy"
]
negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated,(worst quality, low quality:1.3), twins, lowers, jpeg artifacts, low quality"

generate_image_size = (512,512)
# Unitの重みの設定
unit1_weight = 1.0
unit2_weight = 1.6

for image_num in load_image_num:

    # 入力データの加工--------------------------------------------------------------------------------------
    image_name = anotation_data_lists[image_num][-1].replace("\n","")
    image_path = WFLW_images + "/" + image_name
    image = Image.open(image_path)
    orig_image_size_x,orig_image_size_y = image.size #オリジナルの画像データのサイズ（ランドマークの座標変換のときに使う）
    # print("orig_image_size_x,orig_image_size_y",orig_image_size_x,orig_image_size_y)
    bounding_box_data = tuple([int(bdbox) for bdbox in anotation_data_lists[image_num][-11:-7]])
    face_width = bounding_box_data[2]-bounding_box_data[0]
    face_height = bounding_box_data[3]-bounding_box_data[1]

    #顔の中心座標
    face_center_x = float((bounding_box_data[2]+bounding_box_data[0]) / 2)
    face_center_y = float((bounding_box_data[3]+bounding_box_data[1]) / 2)

    # コントロールネット用の画像のクロップとリサイズ
    margin_pix = float((face_width+face_height) / 2 * mergin_per)
    half_long_side = max(face_width,face_height) / 2
    image_size = margin_pix + half_long_side
    crop_coordinates = (
        face_center_x-image_size,
        face_center_y-image_size,
        face_center_x+image_size,
        face_center_y+image_size
    )

    input_control_image = image.crop(crop_coordinates)
    croped_image_size_x,croped_image_size_y = input_control_image.size
    input_control_image = input_control_image.resize(input_control_image_size)
    # print("input_control_image_size",input_control_image.size)
    # input_control_image.show() # controlnet input用の画像の表示#################################

    #scribble用の白黒画像の作成と表示---------------------------------------------------------------------------------
    black_image = Image.new('RGB', input_control_image.size, "black")
    draw = ImageDraw.Draw(black_image)

    orig_landmark_coordinates = anotation_data_lists[image_num][0:196]
    transformed_coordinates_scribble = []
    for landmark_num in range(0, len(orig_landmark_coordinates),2):
        transformed_x = (float(orig_landmark_coordinates[landmark_num])-crop_coordinates[0]) * (input_control_image_size[0] / croped_image_size_x)
        transformed_y = (float(orig_landmark_coordinates[landmark_num+1])-crop_coordinates[1]) * (input_control_image_size[1] / croped_image_size_y)
        transformed_coordinates_scribble.append((transformed_x,transformed_y))

    # 使用する点と線の表示
    draw.line(transformed_coordinates_scribble[0:33], fill="white",width=1) # 輪郭
    # draw.line(transformed_coordinates_scribble[33:38], fill="white",width=1) # 左眉
    # draw.line(transformed_coordinates_scribble[42:47], fill="white",width=1) # 右眉
    draw.line(transformed_coordinates_scribble[51:55], fill="white",width=1) # 鼻筋
    # draw.line(transformed_coordinates_scribble[55:60], fill="white",width=1) # 鼻下
    # draw.line(transformed_coordinates_scribble[88:96], fill="white",width=1) # 口
    # draw.line([transformed_coordinates_scribble[88],transformed_coordinates_scribble[95]], fill="white",width=1) # 口
    # print("black_image.size",black_image.size)
    # black_image.show()

    # イラスト生成用のコード------------------------------------------------------------------------------

    # 生成画像のランドマーク用意
    genimage_keypoints_x = [keypoint[0]*(generate_image_size[0] / input_control_image_size[0]) for keypoint in transformed_coordinates_scribble]
    genimage_keypoints_y = [keypoint[1]*(generate_image_size[1] / input_control_image_size[1]) for keypoint in transformed_coordinates_scribble]
    x_np = np.array(genimage_keypoints_x)
    y_np = np.array(genimage_keypoints_y)
    # print(x_np)
    # print(y_np)

    unit1 = webuiapi.ControlNetUnit(
        input_image=black_image,
        # module='black_image',
        model='control_v11p_sd15_scribble_fp16 [4e6af23e]',
        weight=unit1_weight
    )

    unit2 = webuiapi.ControlNetUnit(
        input_image=input_control_image,
        # input_image=black_image_openpose,
        module='openpose_faceonly',
        model='control_v11p_sd15_openpose [cab727d4]',
        weight=unit2_weight
    )
    # ig,ax = plt.subplots(1)
    r = api.controlnet_detect(images=[input_control_image], module='openpose_faceonly')
    # ax.imshow(r.image)

    for prompt_num,prompt in enumerate(prompts):
        r = api.txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=generate_image_size[0],
            height=generate_image_size[1],
            steps=38,
            controlnet_units=[unit2,]
        )

        # 画像とアノテーションデータの保存-------------------------------------------------------------------
        # 生成された画像の保存
        # 画像の名前、モデル、gitのタグ、日付、プロンプトの番号
        result_current_model = api.util_get_current_model()
        current_model = result_current_model.split("_")[0]
        latest_tagname = get_latest_tagname()
        current_datetime = get_current_now_datetime()
        save_genimg_name = f"{current_model}_git{latest_tagname}_{current_datetime}_{prompt_num}.jpg"
        # print(save_genimg_name)
        r.image.save(f"{image_save_path}/{save_genimg_name}")

        # アノテーションデータの保存
        csv_write_data_list = [f"{save_genimg_name}",f"{image_name}",f"{result_current_model}"]
        for keypoint in range(98):
            csv_write_data_list.append(genimage_keypoints_x[keypoint])
            csv_write_data_list.append(genimage_keypoints_y[keypoint])
        print(csv_write_data_list)
        # アノテーションデータの保存
        with open(f"{annotation_save_path}/annotations.csv","a") as f:
            writer = csv.writer(f)
            writer.writerow(csv_write_data_list)
