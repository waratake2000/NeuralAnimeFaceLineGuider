import imgaug.augmenters as iaa

# 左右反転用
def augmentation_1():
    return iaa.Sequential([
        iaa.Fliplr(1.0)
    ])

def augmentation_2():
    return iaa.Sequential([
        iaa.Fliplr(0.5),  # 50%の確率で左右反転
        iaa.Affine(
            rotate=(-80, 80),# 右、左回りに80度回転させる
            scale={"x": (0.5, 1.2), "y": (0.5, 1.2)},# x軸y軸それぞれずらす
        )
    ])

augmentation_list = [augmentation_1, augmentation_2]
