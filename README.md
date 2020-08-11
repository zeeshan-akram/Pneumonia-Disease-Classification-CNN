# Pneumonia Disease Classification: Project 
I created this model to classify pneumonia disease. I did this project on Kaggle where I used TPU (tensorflow processing unit) which is 25% faster than GPU.
* Data from [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
* Configured TPU.
* Preform Preprocessing on data.
* Get image data for modeling.
* Build model using transfer learning.
* Train Schedule (custom learning rate).
* Tain model with custom weights because of imbalnce data.
* Make predictions with and without test time augmentation.
## Code and Resources Used
**Python Version:** 3.7
**Packages:** numpy, pandas, kaggle_datasets, tensorflow, keras, matplotlib, efficientnet, sklearn, seaborn, re, math, cv2, PIL, os.
**Learn from Notebooks:** [Kaggle Notebook](https://www.kaggle.com/agentauers/incredible-tpus-finetune-effnetb0-b6-at-once).
**Handling imbalance data:** [tensorflow tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
## Configuration
In this step i configured tpu and set constant values of batch size, image size, number of epochs and test time augmentation size.
## Preprocessing
In preprocess prepare files path for training, validation and testing. label the files and perform train test split. <br>
Also checked for imbalance data.<br>
**Normal Cases:** 25% <br>
**Pneumonia Cases:** 75%<br>
Data was imbalance. I tried to fix by initializing model with custom weights.
## Get image data for modeling.
