# Pneumonia Disease Classification: Project Overview
I created this model to classify pneumonia disease. I did this project on Kaggle where I used TPU (tensorflow processing unit) which is 25% faster than GPU.
* Data from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
* Configured TPU.
* Preform Preprocessing on data.
* Get images data for modeling.
* Build model using transfer learning.
* Train model.
* Made predictions with and without test time augmentation.
## Code and Resources Used
**Python Version:** 3.7 <br>
**Packages:** numpy, pandas, kaggle_datasets, tensorflow, keras, matplotlib, efficientnet, sklearn, seaborn, re, math, cv2, PIL, os.<br>
**Learn from Notebooks:** [Kaggle Notebook](https://www.kaggle.com/agentauers/incredible-tpus-finetune-effnetb0-b6-at-once). <br>
**Handling imbalance data:** [tensorflow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data). <br>
**EfficientNet B Model:** [Google AI Blog](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html#:~:text=EfficientNet%3A%20Improving%20Accuracy%20and%20Efficiency%20through%20AutoML%20and%20Model%20Scaling,-Wednesday%2C%20May%2029&text=Powered%20by%20this%20novel%20scaling,efficiency%20(smaller%20and%20faster)).
## Configuration
In this step I configured tpu and set constant values of batch size, image size, number of epochs and test time augmentation size.
## Preview
**Normal:**<br>
![loading or Error](https://github.com/zeeshan-akram/Pneumonia-disease-detection-deep-learning/blob/master/normal.png)<br>
**Pneumonia:**<br>
![loading or Error](https://github.com/zeeshan-akram/Pneumonia-disease-detection-deep-learning/blob/master/pneumonia.png)
## Preprocessing
In preprocess prepare files path for training, validation and testing. label the files and perform train test split. <br>
Also checked for imbalance data.<br>
**Normal Cases:** 25% <br>
**Pneumonia Cases:** 75%<br>
Data was imbalance. I tried to fix by initializing model with custom weights.
## Get images data for modeling.
In this step I retrive images from their paths and convert those images data into tensors. I also perform image augmentation.<br>
**Image Augmentations:**<br>
* Rotation.
* Shear.
* Zoom.
* Shifting.
* Random Brightness.
* Random Flip Left Right.
## Modeling
Create model using transfer learning tecnique. Model I used is EfficientNET B6. EfficientNet are rethinking and compound scaling models. Which constantly improve accuracy.<br>
As data was imbalance so accuracy metrics won't work here. I used different metrics which specifically for imbalance data. Global Average Pooling and Dense layer sigmoid As output layer.<br>
**Metrics:**<br>
* True Positives
* True Negatives
* False Positives
* False Negatives
* Precision
* Recall
* Binary Accuracy
## Model Training
Trained model with custom learning rate schedule and weights.<br>
**Model Parameters:**<br>
Total params: 40,962,441<br>
Trainable params: 40,738,009<br>
Non-trainable params: 224,432<br>
**Training Result:**<br>
Recall: 0.9542<br> 
True Negative: 1026.0000<br>
True Positive: 2896.0000 <br>
False Positive: 35.0000<br> 
Precision: 0.9881 <br> 
False Negative: 139.0000<br>
Loss: 0.1133 <br>
Accuracy: 0.9575 <br> 
**Validation Results:**<br>
Validate Recall: 0.9744<br> 
Validate True Negative: 257.0000 <br> 
Validate False Positive: 10.0000 <br>
Validate Precision: 0.9870 <br>
Validate False Negative: 20.0000 <br>
Validate Loss: 0.0902<br>
Validate True Positive: 760.0000 <br> 
Validate Accuracy: 0.9713 <br>
## Predictions
Made predictions with and without test time augmentation on test files. <br>
**Accuracy without TTA:** 88.30% <br>
**Accuracy with TTA:** 87.34%
