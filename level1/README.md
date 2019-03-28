# 第一类验证码

## 问题介绍

数字四则运算，有噪点干扰，输出计算结果。

![第一类验证码](https://github.com/czczup/Captcha-Recognition/blob/master/docs/problem1.png?raw=true)

## 问题分析

- 验证码不等长，少则5个符号，多则8个符号
- 共有10种数字和3种运算符
- 背景干扰线的颜色和字符有细微差别

## 模型结构

### 彩图

![第一类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net1.png?raw=true)

### 黑白图

![第一类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net1_.png?raw=true)

### 介绍

在该类验证码中，存在10种数字和3种运算符需要识别，为一个13分类问题。在经过降噪处理后，图像的复杂度大大降低，噪点和干扰线几乎完全消除，采用简单的三层全连接神经网络即可达到**99.98%的测试集准确率**。在该网络模型中，dropout的值为0.7，学习率为0.001，使用tanh作为激活函数，使用Adam算法进行训练。

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

  从百度网盘下载第一类验证码数据，分别将训练集和测试集解压放置在：

  - 训练集：image/train
  - 测试集：image/test

- **步骤二：去噪二值化**

  ```
  python denoise_opencv.py
  ```

  去噪后结果如图：

  ![去噪](https://github.com/czczup/Captcha-Recognition/blob/master/docs/denoise1.png?raw=true)

- **步骤三：切割**

  ```
  python cut_captcha.py
  ```

  用[**投影法**](https://blog.csdn.net/wx7788250/article/details/60139109)将验证码切成单字

- **步骤四：生成TFRecord**

  ```
  python generate_tfrecord.py
  ```

- **步骤五：训练**

  ```
  python train.py
  ```

- **步骤六：测试**

  ```
  python test.py
  ```

- **步骤七：计算准确率**

  ```
  python accuracy_calculate.py
  ```

- **额外：查看TensorBoard**

  ```
  tensorboard --logdir=logs
  ```

  

