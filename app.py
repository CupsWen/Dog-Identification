# coding:utf-8
# 导入相关函数包
import os

import numpy as np
import tensorflow as tf
import vgg16 as vgg16
import utils as utils

img_path = input('Input the path and image name:')
img_ready = utils.load_image(img_path)
labels = ['dhole', 'Doberman', 'Yorkshire_terrier','Weimaraner','Bernese_mountain_dog','Pembroke','German_short','English_foxhound','American_Staffordshire_terrier','miniature_pinscher','borzoi','redbone','briard','miniature_poodle','kelpie','schipperke','Bouvier_des_Flandres','Dandie_Dinmont','miniature_schnauzer','Tibetan_mastiff','affenpinscher','Irish_water_spaniel','Scotch_terrier','vizsla','Rottweiler','Blenheim_spaniel','standard_schnauzer','Kerry_blue_terrier','Cardigan','Chihuahua','German_shepherd','Eskimo_dog','whippet','African_hunting_dog','Sussex_spaniel','Shetland_sheepdog','Scottish_deerhound','beagle','bluetick','curly','Appenzeller','Airedale','toy_poodle','Rhodesian_ridgeback','Leonberg','English_springer','chow','boxer','standard_poodle','cairn','English_setter','Siberian_husky','keeshond','Sealyham_terrier','Maltese_dog','Welsh_springer_spaniel','malamute','wire','Great_Dane','Irish_wolfhound','kuvasz','groenendael','bull_mastiff','golden_retriever','Greater_Swiss_Mountain_dog','toy_terrier','Brabancon_griffon','Norwich_terrier','Samoyed','otterhound','giant_schnauzer','Norwegian_elkhound','Great_Pyrenees','papillon','flat','Italian_greyhound','clumber','Bedlington_terrier','Pomeranian','Mexican_hairless','black','Lakeland_terrier','Pekinese','basenji','komondor','silky_terrier','West_Highland_white_terrier','Chesapeake_Bay_retriever','French_bulldog','Newfoundland','pug','Boston_bull','Norfolk_terrier','Saint_Bernard','EntleBucher','Border_terrier','Brittany_spaniel','malinois','Border_collie','Gordon_setter','Australian_terrier','Labrador_retriever','basset','Tibetan_terrier','Staffordshire_bullterrier','Irish_setter','Japanese_spaniel','dingo','Irish_terrier','Saluki','Shih','collie','Old_English_sheepdog','Ibizan_hound','Afghan_hound','soft','Walker_hound','Lhasa','Walker_hound','Lhasa']
print(len(labels))

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

with tf.Session(config=gpuConfig) as sess:
    # 给输入图片占位
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    # 初始化vgg,读出保存的文件参数
    vgg = vgg16.Vgg16()
    # 调用前向传播函数,把待识别图片喂入神经网络
    vgg.forward(images)
    # 运行会话,预测分类结果
    probability = sess.run(vgg.prob, feed_dict={images:img_ready})
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:", top5)
    for n, i in enumerate(top5):
        print("n:", n)
        print("i:", i)
        print(i, ":", labels[i])
        print("----", utils.percent(probability[0][i]))

