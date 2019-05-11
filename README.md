# 这是我在GitHub的第一个项目,很是激动.
小白一枚,HelloWorld.

功能:识别狗的种类(120种).

模型训练:数据集用的是斯坦福大学狗子数据集,基础模型是VGG16,正确率在70%左右.

应用方式:python Django Web 应用.

识别过程:two stage ,先YOLO物体检测出狗在图片中的位置(网上的模型),然后VGG16细致划分.
# 使用
1、下载VGG16.npy,VGG16.npy放于项目根目录下,yolov3.weights放于yolo目录下。

vgg16.npy(VGG16模型卷积核和全连接权重偏置)

链接：https://pan.baidu.com/s/1-kzh69q25E5mt_0CxHuHsA 

提取码：m3yo

vgg15.npy(自己训练的全连接权重和偏置)

链接：https://pan.baidu.com/s/1vJvUlZaEMtHJfPh89JMx-A 

提取码：g0th

yolov3.weights(YOLO参数)

链接：https://pan.baidu.com/s/1-huPnb3_FwUQlxbYHtHtXw 

提取码：s8y1

2、安装相关代码包(我看了一下我的代码包如下所示)。

tensorflow,opencv-python,scikit-image,django版本无所谓冲突概率很小(我没碰到).

numpy版本很重要,涉及到VGG16模型数据的加载,最新的numpy不支持这样的加载方式.

tensorflow             1.3.0 

scikit-image           0.15.0 

opencv-python          4.1.0.25 

numpy                  1.16.2  

Django                 2.2.1  

2、控制台输入：
python manage.py runserver


3、结果。

选择识别图片效果图:

![image](https://github.com/CupsWen/Dog-Identification/blob/master/pic/1.png)

加载图片效果图:
![image](https://github.com/CupsWen/Dog-Identification/blob/master/pic/2.png)


YOLO物体检测效果:
![image](https://github.com/CupsWen/Dog-Identification/blob/master/pic/3.png)

VGG16分类效果:
![image](https://github.com/CupsWen/Dog-Identification/blob/master/pic/4.png)

4、项目结构.
```
├── image                           中间图片存储文件
│       ├── data.jpg                上传图片
|       ├── data_.jpg               图片YOLO检查结果缓存
|       ├── data_crop.jpg           根据YOLO检查结果剪切图片,用于VGG16识别其种类
│── mysite                          Web项目文件
│       ├── setting.py              web配置文件
│       ├── urls.py                 路由配置
│       ├── view.py                 路由后的处理逻辑
│       ├── wsgi.py
├── pic                             md显示图片
│       ├── 斯坦福大学狗子种类名称翻译
│       ├── 世界名犬名字翻译
├── template                        html存储文件
│       ├── index.html              
│       ├── vgg.html
│       ├── yolo.html              
│       ├── yolo_vgg.html
├── yolo                            yolo文件
│       ├── yolov3.cfg              描述文件
│       ├── yolov3.txt              分类class
│       ├── yolov3.weights          模型参数
├── manage.py                       django入口文件
├── utils.py                        vgg16工具类
├── vgg15.py                        vgg16分类class
├── vgg15.npy                       全连接存储文件
├── vgg16.py                        vgg16模型加载文件
├── vgg16.npy                       vgg16模型数据
├── yolo_opencv.py                  yolo使用示例文件
```
# 结论
