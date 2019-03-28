# 第四类验证码

## 问题介绍

中文验证码，包含 4 个中文汉字，有噪点干扰，验证方法为要求用户选出 4 个汉字中被旋转 90 度的那一个 （四个汉字从左到右序号为 0,1,2,3，输出被旋转的汉字序号即可） 。

![第四类验证码](https://github.com/czczup/Captcha-Recognition/blob/master/docs/problem4.png?raw=true)

## 问题分析

- 两种汉字，找出旋转了的汉字
- 位置固定，直接切割

## 模型结构

### 彩图

![第四类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net4.png?raw=true)

### 黑白图

![第四类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net4_.png?raw=true)

### 介绍

验证码中仅存在左旋90°一种情况，可以将已旋转和未旋转分别视为一类，因此本问题为一个2分类问题。另外，对两类图像进行旋转，扩充数据。在上图网络中，dropout的值为0.8，学习率为0.001，使用ReLU作为激活函数，使用Adam优化算法进行网络优化，可达到99.98%的测试集准确率。

经过多种尝试，决定使用下图所示卷积神经网络，经过训练与测试算法优化，达到了**99.98%的测试集准确率**。

## 文件目录介绍

- python代码
  - accuracy_calculate.py 用于计算模型准确率
  - conf.py 路径配置
  - cut_captcha.py 切割验证码
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

  从百度网盘下载第四类验证码数据，分别将训练集和测试集解压放置在：

  - 训练集：image/train
  - 测试集：image/test

- **步骤二：切割**

  ```
  python cut_captcha.py
  ```

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

  

