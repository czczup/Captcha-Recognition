# 第二类验证码

## 问题介绍

英文字母+数字验证码，包含 5 个字符，有噪点干扰，文字无旋转形变，验证方法为要求用户输出验证码中的字符，大小写不限（ 为验证方便可统一转为大写 ）。

![第二类验证码](https://github.com/czczup/Captcha-Recognition/blob/master/docs/problem2.png?raw=true)

## 问题分析

- 字符位置比较固定，无粘连，可以直接分割
- 背景干扰线的颜色和字符不同

## 模型结构

### 彩图

![第二类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net2.png?raw=true)

### 黑白图

![第二类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net2_.png?raw=true)

### 介绍

第二类验证码为数字与英文字母，在本问题中，去除人类易混淆的字符后，共有8种数字和24种英文字母需要识别，为一个32分类问题。在该网络模型中，dropout的值为0.5，学习率为0.001，使用ReLU作为激活函数，使用Adam进行优化。

## 文件目录介绍

- python代码
  - accuracy_calculate.py 用于计算模型准确率
  - conf.py 路径配置
  - cut_captcha.py 切割验证码
  - denoise_opencv.py 验证码去噪
  - generate_tfrecord.py 生成tfrecord
  - model.py 模型定义
  - train.py 训练代码
  - test.py 测试代码
  - util.py 包含一个csv读取函数
- 文件夹
  - mappings 训练集标签/测试集标签/预测标签
  - model 生成的模型
  - logs TensorBoard日志文件
  - tfrecord TFRecord文件目录
  - image 数据集
    - train 训练集数据
    - test 测试集数据
    - cut 切割后的数据
    - denoise 降噪后的数据

## 快速开始

- **步骤一：数据放置**

  下载地址：[百度网盘](https://pan.baidu.com/s/1A07EiNpy7e3sXSyaVyDvSA)  提取码：**e6zy**

  从百度网盘下载第二类验证码数据，分别将训练集和测试集解压放置在：

  - 训练集：image/train
  - 测试集：image/test

- **步骤二：去噪并切割**

  ```
  python cut_captcha.py
  ```
去噪后结果如图：

  ![去噪](https://github.com/czczup/Captcha-Recognition/blob/master/docs/denoise2.png?raw=true)

- **步骤三：生成TFRecord**

  ```
  python generate_tfrecord.py
  ```

- **步骤四：训练**

  ```
  python train.py
  ```

- **步骤五：测试**

  ```
  python test.py
  ```

- **步骤六：计算准确率**

  ```
  python accuracy_calculate.py
  ```

- **额外：查看TensorBoard**

  ```
  tensorboard --logdir=logs
  ```

  

