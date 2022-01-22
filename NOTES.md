# 相关论文
# 常用环境配置
## cv2
> conda install -c conda-forge opencv
# 代码说明
## logging配置
## add_mask_all.py
在处理大量数据的场景下，使用`tqdm`了解进度及程序是否卡住很重要
## data_preparation.py
使用方式：
```
cd face_sdk 运行
``` 

功能：

为批量加口罩(add_mask_all.py)准备数据，主要包括以下两类数据

* 人脸图像的关键点信息
* 人脸图像对应的口罩模板信息

原理:
* 人脸检测模型: RetinaFace [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/pdf/1905.00641.pdf)
* 人脸关键点模型: PFLD [PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf)

考量:

* 对人脸检测——>人脸对齐这类流水线操作, 采用生产者-消费者模式进行处理。
* 对批量图像加口罩可能会遇到两种场景: 1)所有图像不分ID, 都放在一个文件夹下；2)图像根据ID存放在不同文件夹下, 所有文件夹在一个根目录下。实现的代码应该可以兼容这两种使用场景。

后续改进:
* 组batch在多卡上进行模型推理进行加速

## dataset_manager.py
功能：

在大多是识别任务中，我们需要不断增加数据迭代模型。这意味着我们需要对每次新采集到的数据进行整理，得到对应的数据信息，并且可以对不同来源的数据进行合并。具体来说，可以拆分成一下功能

* 新来一批数据[data_root——> id_directory——> images( video_directory——> frame_images)]。对这批数据进行整理，输出一个文本文件，没行包括`rel_data_path`及`labelID`。



考量:

* 有的数据是划分成了`train`、`val`和`test`的，有的直接放在一个文件夹下 。
* 我们只维护一个完整的数据集，及各个来源的文本文件，通过文本文件的组合来决定使用哪些数据。

# 人脸数据集
## 训练集
### Webface
* 50W Images
* 10575 ID [1WID]
### Celeb-MS1M-V1
* 328W Images
* 72778 ID [7WID]
## 测试集
### LFW
 Labeled Faces in the Wild：a database of face photographs designed for studying the problem of `unconstrained face recognition`.
 * 13233 Images
 * 5749 ID [6K ID]
 LFW的原始图像都是包含头肩的，一般需要做预处理：1）人脸检测；2）关键点检测；3）对齐。可以使用`FaceX-ZOO/data/files/lfw_face_info.txt`包含的bounding box及landmarks信息和`FaceX-ZOO/test_protocal/lfw/face_cropper/crop_lfw_by_arcface.py`进行处理。
###


