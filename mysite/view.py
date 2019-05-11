import cv2
import os

from PIL import Image

import utils as utils
import numpy as np
import tensorflow as tf
import vgg16 as vgg16

from django.http import HttpResponse
from django.shortcuts import render, render_to_response


def index(request):
    return render(request, "index.html")


labels = ['Rhodesian_ridgeback 猎狮犬', 'Mexican_hairless 墨西哥无毛犬', 'pug 巴哥犬', 'briard 布雷猎犬', 'miniature_schnauzer 小型雪纳瑞犬',
          'Sealyham_terrier 锡利哈姆梗', 'golden_retriever 金毛寻回犬', 'whippet 惠比特犬', 'Norwegian_elkhound 挪威猎鹿犬',
          'Norwich_terrier 挪威梗', 'Lakeland_terrier 湖畔梗', 'miniature_poodle 迷你贵宾犬', 'English_foxhound 英国猎狐犬',
          'miniature_pinscher 迷你杜宾犬', 'Great_Pyrenees 大白熊犬', 'English_springer 英国斯普林格犬', 'Shih-Tzu 西施犬',
          'Siberian_husky 西伯利亚哈士奇', 'basset 短腿猎犬', 'Saint_Bernard  圣伯纳犬', 'malinois 马利诺斯犬', 'Chihuahua 吉娃娃',
          'black-and-tan_coonhound 黑褐猎浣熊犬', 'Tibetan_terrier 西藏梗', 'giant_schnauzer 巨型雪纳瑞犬', 'Saluki 萨路基猎犬',
          'Doberman 杜宾犬', 'Shetland_sheepdog 喜乐蒂牧羊犬', 'Airedale 秋田犬', 'collie 柯利犬',
          'Chesapeake_Bay_retriever 切萨皮克海湾寻回犬', 'dhole  亚洲豺犬', 'Samoyed 萨摩耶犬', 'Weimaraner 威玛犬',
          'Kerry_blue_terrier 凯利蓝梗', 'Brittany_spaniel 塔尼猎犬', 'boxer 拳师犬', 'Bernese_mountain_dog 伯恩山犬',
          'affenpinscher 猴头梗', 'African_hunting_dog 美国猎犬', 'French_bulldog 法国斗牛犬', 'wire-haired_fox_terrier 刚毛猎狐梗',
          'Irish_terrier 爱尔兰梗', 'Blenheim_spaniel 布莱尼姆长耳猎狗', 'German_short-haired_pointer 德国短毛猎犬',
          'German_shepherd 德国牧羊犬', 'bloodhound 寻血猎犬', 'Rottweiler 罗威纳犬', 'keeshond 荷兰毛狮犬', 'papillon 蝴蝶犬',
          'Afghan_hound 阿富汗猎犬', 'Irish_setter 爱尔兰雪橇犬', 'Irish_wolfhound 爱尔兰猎狼犬', 'clumber 矮脚西班牙猎犬',
          'Sussex_spaniel 苏塞克斯猎犬', 'Appenzeller', 'English_setter 英格兰雪达犬', 'Old_English_sheepdog 古代英国牧羊犬',
          'basenji 巴仙吉犬', 'Italian_greyhound 意大利灰狗', 'Pomeranian 博美犬', 'Border_collie 边境牧羊犬', 'EntleBucher',
          'komondor 匈牙利牧羊犬', 'dingo 澳洲野犬', 'Welsh_springer_spaniel 威尔士跳猎犬', 'chow 松狮犬', 'Brabancon_griffon 布鲁塞尔格里枫犬',
          'Australian_terrier 澳大利亚梗', 'otterhound 猎水獭犬', 'Cardigan', 'Boston_bull', 'Yorkshire_terrier 约克夏梗',
          'groenendael 比利时牧羊犬', 'Scottish_deerhound 苏格兰猎鹿犬', 'Dandie_Dinmont 矮脚狄文㹴', 'Tibetan_mastiff 藏獒',
          'Walker_hound 沃克猎犬', 'kelpie 澳大利亚卡尔比犬', 'Ibizan_hound 伊比赞猎犬', 'silky_terrier 丝毛梗', 'bull_mastiff 斗牛獒',
          'Pekinese  狮子狗', 'Bedlington_terrier 贝灵顿梗', 'soft-coated_wheaten_terrier', 'kuvasz 库瓦兹犬',
          'Greater_Swiss_Mountain_dog 大瑞士山地犬', 'Irish_water_spaniel 爱尔兰水猎犬', 'schipperke 西帕基犬', 'Pembroke 彭布罗克犬',
          'Great_Dane 大丹犬', 'West_Highland_white_terrier 西高地白梗', 'Lhasa 拉萨犬', 'cocker_spaniel 美国可卡犬', 'beagle 比格猎犬',
          'Labrador_retriever 拉布拉多寻回犬', 'borzoi 苏俄猎狼犬', 'cairn', 'Leonberg 莱昂贝格犬', 'bluetick  蓝色快狗', 'toy_poodle 玩具贵宾犬',
          'Maltese_dog 马尔济斯犬', 'Japanese_spaniel 日本猎犬', 'Gordon_setter 戈登雪达犬', 'standard_poodle 标准贵宾犬',
          'Border_terrier 博得猎狐犬', 'malamute  阿拉斯加玛拉慕蒂犬', 'redbone  惬意梗', 'Staffordshire_bullterrier 斯塔福郡斗牛梗',
          'Newfoundland  纽芬兰犬', 'Bouvier_des_Flandres 波兰德斯布比野犬', 'standard_schnauzer 标准型雪纳瑞犬', 'toy_terrier 曼彻斯特玩具梗',
          'Scotch_terrier 苏格兰小猎犬', 'flat-coated_retriever 弗莱特寻回犬', 'curly-coated_retriever 卷毛寻回犬', 'vizsla 维希拉猎犬',
          'Norfolk_terrier 诺福克梗', 'Eskimo_dog 爱斯基摩犬', 'American_Staffordshire_terrier 美国斯塔福郡梗']


def detect_image_vgg16(request):
    if request.method == "POST":                        # 请求方法为POST时，进行处理

        myFile = request.FILES.get("myfile", None)      # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")
        img_ready = utils.load_image(myFile)
        with tf.Session() as sess:
            # 给输入图片占位
            images = tf.placeholder(tf.float32, [1, 224, 224, 3])
            # 初始化vgg,读出保存的文件参数
            vgg = vgg16.Vgg16()
            # 调用前向传播函数,把待识别图片喂入神经网络
            vgg.forward(images)
            # 运行会话,预测分类结果
            probability = sess.run(vgg.prob, feed_dict={images: img_ready})
            top5 = np.argsort(probability[0])[-1:-6:-1]
            data = ""
            data += "top5:"
            data += str(top5)
            data += "<br/>"
            for n, i in enumerate(top5):
                data += str(i)
                data += ":"
                data += labels[i]
                data += "----"
                data += str(utils.percent(probability[0][i]))
                data += "<br/>"

        return HttpResponse("<html><center><h1>VGG16物体分类结果展示</h1><body>%s</body></center></html>" % data)



class_path = "yolo/yolov3.txt"
weights_path = "yolo/yolov3.weights"
config_path = "yolo/yolov3.cfg"
locations = []
def detect_image_yolo(request):
    if request.method == "POST":                        # 请求方法为POST时，进行处理
        myFile = request.FILES.get("myfile", None)      # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")

        file = open(os.path.join(os.getcwd(), 'image/data.jpg'), 'wb+')
        for chunk in myFile.chunks():  # 分块写入文件
            file.write(chunk)
        file.close()

        image = cv2.imread("image/data.jpg")
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        classes = None

        # 种类读取
        with open(class_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(weights_path, config_path)
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # 显示识别效果
        locations.clear()
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            if class_ids[i] == 16:
                locations.append([x, y, w, h])
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h),classes,COLORS)
        cv2.imwrite("image/data_.jpg", image)
        print("dogs locations:", locations)
        return render(request, 'yolo.html')


def detect_image_yolo_vgg(request):
    data = ""
    for i in locations:
        img = Image.open("image/data.jpg")
        img = img.crop([i[0], i[1], i[0]+i[2], i[1]+i[3]])
        img.save("image/data_crop.jpg")
        img_ready = utils.load_image("image/data_crop.jpg")
        with tf.Session() as sess:
            # 给输入图片占位
            images = tf.placeholder(tf.float32, [1, 224, 224, 3])
            # 初始化vgg,读出保存的文件参数
            vgg = vgg16.Vgg16()
            # 调用前向传播函数,把待识别图片喂入神经网络
            vgg.forward(images)
            # 运行会话,预测分类结果
            probability = sess.run(vgg.prob, feed_dict={images: img_ready})
            top5 = np.argsort(probability[0])[-1:-2:-1]
            data += str(i)
            data += "--"
            data += labels[top5[0]]
            data += "<br/>"
    return HttpResponse("<html><center><h1>YOLO+VGG16物体检测结果展示</h1> <img src=\"http://127.0.0.1:8000/image/data_.jpg\"/><br/><body>%s</body></center></html>" % data)

# 获取某层输出
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# 画出识别结果
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)