# 第五类验证码

## 问题介绍

中文验证码，包含 4 个中文汉字和 9 个中文单字，有噪点干扰，文字有旋转形变，验证方法为要求用户从 9 个单字中从左到右按顺序选出验证码中的汉字，输出汉字编号。

![第五类验证码](https://github.com/czczup/Captcha-Recognition/blob/master/docs/problem5.png?raw=true)

## 问题分析

- 汉字匹配，仅需判断是否是同一个字，无需识别出具体是什么字
- 位置固定，直接切割

## 模型结构

### 彩图

![第五类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net5.png?raw=true)

### 黑白图

![第五类验证码模型](https://github.com/czczup/Captcha-Recognition/blob/master/docs/net5_.png?raw=true)

### 介绍

第五类验证码为汉字，要求进行相似度匹配并输出编号。使用度量学习中的Siamese Network。经过训练与测试算法优化，最终可达**97.40%的测试集准确率**。在该网络模型中，学习率为0.0001，使用ReLU作全连接层的激活函数，使用Sigmoid作输出层的激活函数。

## 文件目录介绍

- python代码
  - accuracy_calculate.py 用于计算模型准确率
  - conf.py 路径配置
  - generate_batch_data.py 生成孪生网络所需正负样本对
  - generate_tfrecord.py 生成tfrecord
  - model.py 模型定义
  - train.py 训练代码
  - util.py 包含一个csv读取函数
  - test.py 测试代码
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

  从百度网盘下载第五类验证码数据，分别将训练集和测试集解压放置在：

  - 训练集：image/train
  - 测试集：image/test

- **步骤二：生成正负样本对**

  ```
  python generate_batch_data.py
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

  

