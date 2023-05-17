# SLR_for_frb
a SLR project for 2023 frb contest 

### Requirements

- Download and extract **[CSL Dataset](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)**
- Download and install **[PyTorch](https://pytorch.org/)**

### description

deploy文件夹为将生成的tflite模型部署于android手机app的工程。

ResNet(2+1)D_model文件夹是搭建模型的源代码。（具体位于.\deploy\lite\examples\video_classification\android\app中）

### comparison

下面选择了4种有代表性的用于手语识别的模型，我们用相同的训练集和测试集下分别测试其识别准确率并进行对比，结果如下图。除此之外，我们还与加速度方案进行了横向对比。

![image-20230517105010804](https://charles2530.github.io/image/frb_compare_1.png)

![image-20230517105123676](https://charles2530.github.io/image/frb_compare_2.png)

### show

手机上部署识别效果图:

![image-20230517105205839](D:/coding_file/coding_desktop/blog/Charles2530.github.io/myBlog/source/image/frb_mobile.png)

### References

- [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/pdf/1711.09577.pdf)

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/pdf/1801.07455.pdf)
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- [SIGN LANGUAGE RECOGNITION WITH LONG SHORT-TERM MEMORY](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7532884)
- https://github.com/HHTseng/video-classification
- https://github.com/kenshohara/3D-ResNets-PyTorch

- https://github.com/bentrevett/pytorch-seq2seq

