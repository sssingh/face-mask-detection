# Face Mask Detection
Comparitive study of face-mask detection using deep-learning, shallow-learning and hybrid techniques.

<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/facemask-unsplash.jpg?raw=true" width="800" height="300">

## Features
⚡Binary Classification  
⚡Cutsom CNN  
⚡SVM Classifier  
⚡Transfer Learning CNN - ResNet50  
⚡Transfer Learning CNN - MobileNetV2  
⚡Fully Connected Classifier  
⚡Tensorflow/Keras  

## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Solution Approach](#solution-approach)
- [Conclusion](#Conclusion)
- [Further Improvements](#further-improvements)
- [How To Use](#how-to-use)
- [License](#license)
- [Get in touch](#get-in-touch)
- [Credits](#credits)

## Introduction
Face masks have proven to be very effective against COVID-19 spread. However, despite facemasks being a very effective preventive measure, society has apparent resistance. However, the seriousness of the COVID-19 spread, authorities have started monitoring and enforcing the usage of facemasks in public places in many countries. But it's almost impossible to do this manually. In recent years, the rise of machine-learning/deep-learning techniques (ML/DL) presents a huge opportunity to fight the pandemic through cutting-edge technology. This project aims to implement, test, compare multiple ML/DL models for automatic facemask detection. 

*This was my thesis for M.S. (Data Science) program. If you are interested, you can access the complete thesis report from [here](https://drive.google.com/file/d/12uLJ9UPAKGBc2Pv160RYt4SEJ5NS4_ES/view?usp=sharing) and thesis defense presentation from [here](https://docs.google.com/presentation/d/1vzMyfzwsXxFYMCvwD3VAkKmPKQ6Py9YF/edit?usp=sharing&ouid=117056812088365898221&rtpof=true&sd=true)*

## Objective
The project aims to design, implement and compare the multitude of machine learning models utilizing modern ML and DL techniques to discriminate between the images of human faces wearing or not wearing facemasks. Following are the main objectives:
- To process the input raw images to get them ready for the model building
  - Upscale (increase image size) and downscale (decrease image size) the raw input images to ensure that they are all of the exact sizes
  - Ensure that there is no imbalance between the number of images in each class (no mask & mask). Balance the dataset by generating additional input images
- Use image augmentation to generate an additional set of images from the original set of input images to increase the volume and variety of the training data.
- To implement and train pure DL architectures from the ground up (i.e., no pre-trained network) to classify images of the people wearing or not wearing facemasks.
- To implement and train pure DL architectures by leveraging the transfer-learning to classify images of the people wearing or not wearing facemasks.
- To implement and train DL architectures from the ground-up but coupled with a classic learning model (e.g., SVM) as the classifier (i.e., hybrid network) to classify images of the people wearing or not wearing facemasks.
- To test, compare and interpret the model performance results based on various performance metrics.

## Dataset
Publicly available [Real-World-Masked-Face-Dataset (RFMD)](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) dataset and [Simulated-World-Masked-Face-Dataset (SFMD)](https://drive.google.com/uc?id=1q0ibaoFVEmXrjlk3-Oyx2oYR8HpVy6jc&export=downloadand) dataset were used for model training, validation, and evaluation. 
- The RFMD dataset contains 90,468 images of real-world human faces not wearing any facemask and 5,017 images of faces wearing facemasks. <br><br>
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/RFMD_sample.png?raw=true">
<br><br>
- The SFMD dataset contains 690 faces stimulated to appear as wearing facemasks and 686 images of faces not wearing any facemask. Figure 8 shows the sample mask and no mask images from the RFMD dataset. 
<br><br>
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/SFMD_sample.png?raw=true">
<br><br>
The raw and fully processed dataset (data.zip, 1.5GB zip file) can be downloaded from  [here](https://drive.google.com/file/d/1THtv0JKDMKhkGgo-W7c-EtZGZ4MdnDgs/view?usp=sharing)
The data.zip contains the main parent folder 'data.' Inside the 'data' folder, there are three sub-folders...<br> 
1. **raw_dataset**: This is the original dataset taken from publicly available RFMD and SFMD datasets. Original images from RFMD and SFMD are combined to create raw_dataset. <br>
2. **flatten_raw_dataset**: This is the raw_dataset after flattening individual sub-folders within the raw_dataset. <br>
3. **processed_dataset**: This is the processed dataset. This dataset is used for model training, validation, and testing.

## Evaluation Metrics
The metrics below were recorded for each of the six models we build during training and evaluation phases...
* Number of Model Parameters
* Model Size (Footprint)
* Training Time
* Prediction Speed/Time
* Accuracy
* Precision
* Recall
* Area Under the Curve and ROC (AUC-ROC)

## Solution Approach

### Data Preprocessing
Before we start building the models, the raw data needs to be preprocessed to get it ready for modeling. Therefore, the below listed preprocessing was performed...
1. The images in the original raw datasets were stored in a multilevel directory structure. Datasets are required to be preprocessed to flatten the directory structure and move both images with masks and images without masks in their respective directory.
2. The images in the original raw datasets were available in different sizes and resolutions. Images were resized to 224x224 size to make them of the same size. 224x224 size was chosen because most of the pre-trained CNN networks expect image inputs in this size.
3. After the above two steps, we end up with a dataset with a heavy class imbalance where the number of real images without masks overpowers the number of real images with masks. To balance the dataset...
    * All real images with mask (5,017 images) and simulated images with mask (690 images) were combined to form a complete masked image dataset of 5,707 images.
    * Out of 90,468 real images without the mask, 5,017 images were chosen randomly and combined with all the simulated images without a mask (686 images) to form a complete "nomask" dataset of 5,703 images.
4. Random variation in images was introduced via image augmentation (rotation, horizontal flip, image shear, zoom in, zoom out, etc.) An important point to note here is that the augmented images were not stored as part of the training dataset, but they were created on the fly during the model training and then discarded. If the model must be trained again, then another set of totally different augmented images would be generated for training.

### Model building
Various models were built using transfer learning (ResNet50, MobileNetV2, etc.) and from scratch. Under both of these categories, further sub-configuration of the model such as a _pure_ network (CNN as a base with fully connected layer (FC) as a classifier layer), a _hybrid_ network (CNN as a feature extractor and shallow learning model such as an SVM is used as a classifier) were implemented.
<br><br>
#### 1. Transfer learning - ResNet50 & SVM Classifier (Hybrid)
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/rn50svm.png?raw=true" alt="">

#### 2. Transfer learning - MobileNetV2 & SVM Classifier (Hybrid)
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/mnv2svm.png?raw=true" alt="">

#### 3. Transfer learning - ResNet50 & FC Classifier (Pure)
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/rn50fc.png?raw=true" alt="">

#### 4. Transfer learning - MobileNetV2 & FC Classifier (Pure)
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/mnv2svm.png?raw=true" alt="">

#### 5. Custom Built - CNN & FC Classifier
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/rn50fc.png?raw=true" alt="">

#### 6. Custom Built - CNN & SVM Classifier
This network uses the same custom CNN architecture as above but instead of a fully-connected classifier, it uses SVM.

## Conclusion
The implemented models were compared and ranked based on the evaluations metrics recorded during the model training and evaluation phases. Models are ranked from 1 to 6 (1 being best and 6 being the worst) against each of the evaluation metrics recorded and based on that final conclusion is drawn.
<br><br>
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/eval_results.png?raw=true" alt="">
<br><br>
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/conclusion1.png?raw=true" alt="">
<br><br>
<img src="https://github.com/sssingh/face-mask-detection/blob/master/images/conclusion2.png?raw=true" alt="">

## Further Improvements
Following are some of the suggested improvements that one can experiment with…
* Custom CNN architecture can be improved by adding a few more convolution blocks at the beginning of the network.
* Initial layer's Conv2D filter-sizes can be kept larger than 3x3 to increase the feature space and allow the model to learn even more fine-grained patterns in the input images.
* MaxPooling from initial layers can be removed to increase the feature space of input images further.
* Shallow learning classifiers other than SVM can be hooked at the end of Custom CNN for image classifications. XGBoost [50] is one of the latest entrants that can be experimented with instead of an SVM classifier.

## How To Use
There are two notebooks provided as part of this repo...
1. **Data Preprocessing.ipynb**: This notebook contains functionality for data preprocessing. The raw data is first flattened, images are re-scaled, and then the data set is balanced to produce a preprocessed dataset used for modeling. This note is required to be run only once on the raw dataset. Since the preprocessed dataset is included in data.zip, there is no need to rerun this notebook.
2. **Model Implementation.ipynb**: This notebook contains functionality for building, training, and evaluating the models. Notebook expects a _data_ folder to be present as the notebook is kept in the exact location. The _data_ folder is expected to contain _processed_dataset_ folder. The best way is to copy the data.zip where the notebook is kept and unzip it. Once the _data_ folder is created, the notebook should be run sequentially from top to bottom. Notebook uses `TensorFlow` as backend and `Keras` frontend for deep-learning models.

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Get in touch
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/sssingh)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/_sssingh)
[![website](https://img.shields.io/badge/website-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://datamatrix-ml.com/)


## Credits
- Title photo by [Adam Nieścioruk on unsplash](https://unsplash.com/@adamsky1973?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

[Back To The Top](#face-mask-detection)
