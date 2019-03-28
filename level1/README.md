# 第一类验证码

## 背景介绍

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