# coding:utf-8
# 导入相关函数库
from skimage import io, transform
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False    # 正常显示正负号

def load_image(path):
    '''
    对图像进行预处理,并
    显示 Original Picture,Centre Picture",Resize Picture
    :param path: 待识别图片路径
    :return: 处理后的图像
    '''
    img = io.imread(path)
    img = img / 255.0
    short_edge = min(img.shape[:2]) 
    y = (img.shape[0] - short_edge) // 2
    x = (img.shape[1] - short_edge) // 2
    crop_img = img[y:y+short_edge, x:x+short_edge]
    re_img = transform.resize(crop_img, (224, 224))
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

def load_image_crop(path):
    '''
    对图像进行预处理,并
    显示 Original Picture,Centre Picture",Resize Picture
    :param path: 待识别图片路径
    :return: 处理后的图像
    '''
    img = io.imread(path)
    img = img / 255.0
    short_edge = min(img.shape[:2])
    y = (img.shape[0] - short_edge) // 2
    x = (img.shape[1] - short_edge) // 2
    crop_img = img[y:y+short_edge, x:x+short_edge]
    re_img = transform.resize(crop_img, (224, 224))
    img_ready = re_img.reshape((1, 224, 224, 3))

    return img_ready

def percent(value):
    '''
    获取百分数表示
    :param value:小数
    :return: 百分数表示
    '''
    return '%.2f%%' % (value * 100)

if __name__ == '__main__':
    f = open("test.txt", "r")
    contents = f.readlines()
    f.close()
    data = list()
    for content in contents:
        raw_data = content.split("/")
        if raw_data[0] not in data:
            data.append(raw_data[0])
            print(raw_data[0])
    print(data)

