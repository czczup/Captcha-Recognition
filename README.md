# 深度学习识别各类验证码

## 背景介绍

- 来源：第九届中国大学生服务外包创新创业大赛赛题**A16-验证码识别**
- 整体背景：验证码识别是计算机与大数据领域一个非常重要的应用领域，其中包含了图像处理、机器视觉、模式识别以及人工智能等诸多前沿技术，这些同样也是大数据未来发展所必须的核心技术。本赛题以互联网中常见的验证码图片为样本，旨在通过此次竞赛，检验参赛者的图像识别能力，提高对计算机技术、算法模型的认识和应用能力，激发在图像处理、机器学习等领域的创新能力。
- 公司背景：**浪潮**卓数大数据产业发展有限公司作为浪潮集团旗下的大数据板块，致力于成为数据资源提供商、数据资产运营商和数据交易服务商，以大数据时代的“数商”为发展目标，促进政府、企业和个人实现从互联网化向社会化，乃至数据社会化的转变。

## 项目说明

### 问题说明

提供 5 类验证码图片，难度依次递增，每一类验证码提供 1 万个训练样本，最后通过另外 5000 个测试样本的识别率评分。

### 第 1 类

数字四则运算，有噪点干扰，输出计算结果。

![captcha1](https://github.com/czczup/Captcha-Recognition/blob/master/docs/captcha1.png?raw=true)

### 第 2 类

英文字母+数字验证码，包含 5 个字符，有噪点干扰，文字无旋转形变，验证方法为要求用户输出验证码中的字符，大小写不限（ 为验证方便可统一转为大写 ）。

![captcha2](https://github.com/czczup/Captcha-Recognition/blob/master/docs/captcha2.png?raw=true)

### 第 3 类

英文字母+数字验证码，包含 4 个字符，有噪点干扰，文字有旋转形变，验证方法为要求用户输出验证码中的字符。

![captcha3](https://github.com/czczup/Captcha-Recognition/blob/master/docs/captcha3.png?raw=true)

### 第 4 类

中文验证码，包含 4 个中文汉字，有噪点干扰，验证方法为要求用户选出 4 个汉字中被旋转 90 度的那一个 （四个汉字从左到右序号为 0,1,2,3，输出被旋转的汉字序号即可） 。

![captcha4](https://github.com/czczup/Captcha-Recognition/blob/master/docs/captcha4.png?raw=true)

### 第 5 类

中文验证码，包含 4 个中文汉字和 9 个中文单字，有噪点干扰，文字有旋转形变，验证方法为要求用户从 9 个单字中从左到右按顺序选出验证码中的汉字，输出汉字。

![captcha5](https://github.com/czczup/Captcha-Recognition/blob/master/docs/captcha5.png?raw=true)

## 快速开始

- **步骤一：环境准备**

  Python 3.6、TensorFlow、Pillow、OpenCV

  ```
  pip install tensorflow-gpu==1.8
  pip install opencv-python
  pip install pillow
  ```

- **步骤二：准备数据集**

  下载地址：[百度网盘](https://pan.baidu.com/s/1A07EiNpy7e3sXSyaVyDvSA) 

  提取码：**e6zy**

  放置位置：

  - 训练集：level_/image/train
  - 测试集：level_/image/test

- **步骤三：选择验证码类型**

  - [第一类：数字四则运算，有噪点干扰，输出计算结果](https://github.com/czczup/Captcha-Recognition/tree/master/level1)

  - [第二类：数字、英文，有噪点干扰，文字无旋转形变，输出数字、英文](https://github.com/czczup/Captcha-Recognition/tree/master/level2)

  - [第三类：数字、英文，有噪点干扰，文字有旋转形变，输出数字、英文](https://github.com/czczup/Captcha-Recognition/tree/master/level3)

  - [第四类：汉字，有噪点干扰，文字无旋转形变，输出汉字](https://github.com/czczup/Captcha-Recognition/tree/master/level4)

  - [第五类：汉字，有噪点干扰，文字有旋转形变，输出汉字](https://github.com/czczup/Captcha-Recognition/tree/master/level5)

### 最后

当时做这个项目的时候是第一次接触深度学习，看着吴恩达和炼数成金的视频课边学边做，真是特别有意思的一段时光。现在回头来看，当时设计的每类验证码识别模型都不同，准确率主要靠调参，确实挺ugly的。还记得答辩的时候评委问我，能不能做一个万能验证码识别模型，可惜当时学识浅陋，我答的“不行”，哈哈。

通过这次竞赛，成功让我入门了深度学习，特别感谢浪潮集团出的题和提供的奖金。当年这题获奖的名额超多的，浪潮真的财大气粗，若要参加服务外包竞赛，浪潮的题目超级推荐。验证码识别是特别好的深度学习入门案例，希望本仓库能给刚入门的你提供一些帮助。