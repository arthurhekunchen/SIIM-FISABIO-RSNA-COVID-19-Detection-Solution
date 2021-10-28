# Kaggle SIIM-FISABIO-RSNA COVID-19 Detection 30th Solution(Enhanced version)

## INTRODUCTION
1. The contestants construct computer vision models to diagnose whether a patient has COVID-19 by X-ray chest radiograph and to localize the infected area of the lung.
2. This competition is a fusion of two models, "Image Classification" and "Target Detection", to get the final result. We can try the idea of "EffecientNet + YOLOv5" to accomplish this.

## DATASET
1. The data size is very large (approximately 100G), but there are only more than 6000 training data. The reason is that the data uses a special format for medical imaging DCM, each image takes up more space.
2. After a brief preview of the data, we found that the data was divided into two levels, study_level and image_level.
   Study represents one examination and image represents one image. A single examination may contain multiple images (the reason may be poorly positioned, unclear imaging, etc.). The training set contains a total of 6054 examinations and 6334 images. 
3. A study may contain only one of the following four results: "negative", "typical", "indeterminate", "atypical".
   Target detection is for image_level and has only one target: opacity (lung opacity), but a single image can have multiple opacities.

## FINAL SOLUTION SUMMARY
1. Using EfficientnetV2-L-in21k for the 4-classification task, after completing the training with 5 Folds, we can get the PublicLB 0.462 for pure classification.
2. Using Yolov5 with modified structure for target detection, the result can be boosted to PublicLB 0.636 after completing the training with 5 Folds.
3. Using WBF to fuse the Yolov5 with the public CascadeRCNN (https://www.kaggle.com/sreevishnudamodaran/siim-effnetv2-l-cascadercnn-mmdetection-infer), we can improve the result by PublicLB 0.638.
4. For the value of none in target detection, using EfficientnetV2-L-in21k to make a 2-classification task (also 5Fold) to predict the result, we can improve the result by PublicLB 0.644.

## MODEL DETAILS

### C**lassification**: EfficientnetV2-L-in21k

- The pre-trained imagenet weights are from the timm library.
- 512 x 512 (5-fold) image size.
- Joint loss function:[0.5* FocalLoss+ 0.5* BCE].
- A warmup CosineAnnealingLR scheduler is used.
- The activation layer of the model was replaced with Mish activation.

###  **Detection**: YOLOv5

**Yolo - v5l6:** It is trained on a pseudo-label of training data and public test data. The image size used for training is 640 x 640. Some images without bounding boxes (20%) are also included in the training data.

**Yolo - v5x:** It is trained on pseudo-labels of the training data and the public test data. The image size used is the default value of Yolo-v5x, i.e. 640 x 640. The training data also includes some images without bounding boxes (20%).

- Pre-train the backbone model of FasterRCNN with chexpert + chest14
- Train model with siim covid trainset, load weights from rsna checkpoint
- Loss function: FocalLoss
- inference stage: 3TTA of Yolov5 (original, scale 0.83 + hflip, scale 0.67)


## DATA ENHANCEMENT METHODS

- RandomResizedCrop
- ShiftScaleRotate
- HorizontalFlip
- VerticalFlip
- Blur
- CLAHE
- IAASharpen
- IAAEmboss
- RandomBrightnessContrast
- Cutout


## COMPETITION TRICK POINTS
1. Because of the unbalanced sample of competition data, we used FocalLoss in the classification, increasing the score by about 0.02.
2. The amount of data is small. In order to prevent overfitting, we used aux task to increase the difficulty of the training task and increased the score by about 0.05. (https://www.kaggle.com/c/siim-covid19-detection/discussion/240233)
3. The data of this competition is very sensitive to super-reference(lr, loss and aug are all very sensitive), which is more easy to over-fit. Thus, we used optuna to adjust the reference, successfully increased the score by 0.05.
4. For the data of this competition, the fusion of target detection using WBF worked well and resulted in a score improvement of 0.02.
5. In the inference stage, we used "hflip" to deal with "tta" during image classification, increasing the score by 0.01 ~ 0.02.


## METHODS IN VAIN
1. Because of the small number of datasets, we introduced external dataset RSNA data for softlabel, but the score did not improve.
2. We tried to introduce aux task in Yolov5, but the code was very complicated. Probably because of the code logic, the score did not improve, so we did not spend extra time later.


## THINGS DIDN'T GET TO TRY

- Multi-task classification + detection
- Stacking multi-classification models using cnn or lgbm
- Mixing of classification models + shear mixing
- Bimcv + ricord dataset: most of the images in bimcv and ricord are duplicated with siim covid trainset and testet. To avoid data leakage during training, we did not use them.
- Increasing the input size of the classification model to 1024, the CV and public LB scores are almost unchanged from 512.





## 比赛介绍
1. 参赛者通过构建计算机视觉模型通过X-ray胸片诊断患者是否患有COVID-19，并且对肺部感染区域进行定位。
2. 这次比赛是通过 图像分类 + 目标检测 两种模型的融合来获得最终结果,可以尝试EffecientNet + YOLOv5的思路来完成。


## 比赛数据
1. 赛题数据非常大（约100G），但有只有6000多条的训练数据,原因是数据采用的医疗影像的专用格式DCM，每张图片占空间较大。
2. 简单预览数据后，我们会发现，数据分为两个层级，study_level 和 image_level。
Study代表一次检查，image代表一张图片。一次检查可能会包含多张图片（原因可能是拍片位置不好、成像不清楚等等）。训练集一共包含6054次检查和6334个图片。
3. 图像分类是针对study_level的，一个study只可能包含以下四种结果中的一种： “阴性”, “典型”, “不确定”, “非典型”
目标检测是针对image_level的，只有一种目标： opacity（肺部不透明度），但一张图片可以有多处opacity。


## 最终方案概要
1. 用EfficientnetV2-L-in21k做4分类任务，完成5个Fold的训练后，可以得到纯分类的PublicLB 0.462。
2. 用修改过结构的Yolov5做目标检测，完成5个Fold的训练后，可以将结果提升PublicLB 0.636。
3. 用WBF融合刚才的Yolov5和公开的CascadeRCNN（ https://www.kaggle.com/sreevishnudamodaran/siim-effnetv2-l-cascadercnn-mmdetection-infer ），可以将结果提升PublicLB 0.638。
4. 对于目标检测中的none值，用EfficientnetV2-L-in21k做一个2分类任务（同样5Fold）来预测，可以将结果提升PublicLB 0.644。

## 模型细节

### C**lassification**: EfficientnetV2-L-in21k

- 预先训练的imagenet权重来自timm库。
- 512 x 512（5-fold）的图像尺寸。
- 联合损失函数:[0.5* FocalLoss+ 0.5* BCE]。
- 使用了warmup CosineAnnealingLR调度器。
- 该模型的激活层被替换为Mish激活。



###  **Detection**: YOLOv5

**Yolo - v5l6:** 它是在训练数据+公共测试数据的伪标签上训练的。训练时使用的图像大小为640 x 640。训练数据中还包括一些没有边界框的图像（20%）。

**Yolo - v5x:** 它是在训练数据和公共测试数据的伪标签上训练的。使用的图像大小是Yolo-v5x的默认值，即640 x 640。训练数据中还包括一些没有边界框的图像（20%）。

- 用chexpert + chest14预训练FasterRCNN的骨干模型
- 用siim covid trainset训练模型，从rsna checkpoint加载权重
- 损失函数：FocalLoss
- inference阶段：Yolov5的3TTA（原始，比例0.83+hflip，比例0.67）



## 数据增强方法

- RandomResizedCrop
- ShiftScaleRotate
- HorizontalFlip
- VerticalFlip
- Blur
- CLAHE
- IAASharpen
- IAAEmboss
- RandomBrightnessContrast
- Cutout




## 比赛Trick点
1. 比赛数据样本不均衡，在分类中使用FocalLoss，涨分0.02左右。
2. 数据较少，为了防止过拟合，使用aux task增加训练任务的难度，涨分0.05左右。（ https://www.kaggle.com/c/siim-covid19-detection/discussion/240233 ）
3. 这次比赛对超参非常敏感， lr loss aug 都很敏感，容易过拟合，使用optuna调参，涨分0.05。
4. WBF做目标检测融合效果很好，涨分0.02。
5. 在推理阶段，图像分类使用hflip做tta，涨分0.01~0.02。



## 尝试后，没有效果的
1. 因为数据集数量少，于是引入外部数据集RSNA数据做softlabel，但没有涨分。
2. 在Yolov5中引入aux task，代码很复杂，可能是因为代码逻辑的问题，没有涨分，但后面也没有再花时间。



## 我们没有来的及尝试的事情

- 多任务分类+检测
- 使用cnn或lgbm堆叠多分类模型
- 分类模型的混合+剪切混合
- Bimcv + ricord数据集：bimcv和ricord中的大部分图像与siim covid trainset和testet重复。为了避免训练时的数据泄露，我没有使用它们。
- 将分类模型的输入大小增加到1024，CV和公共LB的得分与512几乎没有变化。

