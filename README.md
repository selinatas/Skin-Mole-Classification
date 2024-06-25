# Skin Mole Classification Web Application 🏥
Skin Moles: Benign vs Malignant (Melanoma) Classification using ISIC 2019 dataset with 9,522 images for Samsung Innovation Campus.
Upload your skin mole picture to understand if it is malignant or not! 🤓

## Motivation :fire:
Skin cancer, particularly melanoma, is one of the most serious types of skin cancer. Early detection and accurate diagnosis are crucial for effective treatment and improving patient outcomes. However, distinguishing between benign moles and malignant melanoma can be challenging even for experienced dermatologists. Leveraging advancements in deep learning and computer vision, this project aims to develop an automated system for classifying skin moles as benign or malignant using the ISIC 2019 dataset.

## Data Preprocessing 📊

* Training Transforms:

  ** Apply random shifts, scales, and rotations.
  
  ** Apply RGB shifts, brightness/contrast adjustments, and multiplicative noise.
  
  ** Normalize images and convert them to tensors.

* Test Transforms:
  
  ** Normalize images and convert them to tensors.

New directories for train validation and test sets are created and the images seperated randomly with the ratios 70%:20%:10% for train, validation and test sets respectively.

## Utilized Deep Learning Algorithms :gear:

* ResNet-50
* DenseNet-121
* Custom CNN model - the architecture is visualized below

  <img width = "250" src="https://github.com/selinatas/Skin-Mole-Classification/assets/110598211/a8aed045-6a2f-4d51-9e7f-d6c84af05906" alt="Custom CNN Architecture" width="400">

## Results 📑

![model_test_results](https://github.com/selinatas/Skin-Mole-Classification/assets/110598211/437fe878-e7e6-4790-959d-ef00811d6dc2)

## Resources 🧱

12GB Nvidia RTX 4080 GPU

## Web Application using Streamlit UI 🖥️:

The screenshots from the web application given below.
Just upload you skin mole photo and select your model, code will do the rest!!

<img width="500" alt="NV1" src="https://github.com/selinatas/Skin-Mole-Classification/assets/110598211/6d492332-0061-4c86-8912-782e00f295e2">
<img width="500" alt="label_mel" src="https://github.com/selinatas/Skin-Mole-Classification/assets/110598211/c7a6d67b-e00c-4a7c-997c-afeeb0ac9c36">

## Code 📖
* dataset.py:
  Responsible for preparing the dataset in a format suitable for deep learning model training, including augmentation, normalization, and creating data loaders for efficient data handling.

* main.py:
  Serves as the central script for model training, evaluation, and results visualization in the project.

* app.py:
  Provides a user-friendly interface for classifying skin moles using pre-trained deep learning models, making it accessible for users to interact with the models without needing to     
  understand the underlying code or architecture. ONLY NEED TO RUN THIS CODE!


## How to run the application ❔

* Make sure that you download the necessary libraries with pip install torch torchvision pandas matplotlib streamlit pillow albumentations opencv-python
* Download the dataset from kaggle https://www.kaggle.com/datasets/adisongoh/skin-moles-benign-vs-malignant-melanoma-isic19
  and copy the label_NV and label_MEL files under a empty file called dataset. This part is important, the file structure must be like the below!!
  <img width="172" alt="image" src="https://github.com/selinatas/Skin-Mole-Classification/assets/110598211/cbdde9cd-8983-4ecb-83c2-e4894d76fa8e">
 
* type "streamlit run app.py" in terminal

 Enjoy!

 ## Author 👤

* Selin Ataş



