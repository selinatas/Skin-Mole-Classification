import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import os
import shutil
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob



####################################################
#       Define Variables
####################################################
dataset_path = "dataset" 
output_path = "images"  
batch_size = 64

train_transforms = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_path = os.path.join(output_path, "train")
val_path = os.path.join(output_path, "validation")
test_path = os.path.join(output_path, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

for label_dir in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label_dir)
    if os.path.isdir(label_path):
        os.makedirs(os.path.join(train_path, label_dir), exist_ok=True)
        os.makedirs(os.path.join(val_path, label_dir), exist_ok=True)
        os.makedirs(os.path.join(test_path, label_dir), exist_ok=True)

        images = [img for img in os.listdir(label_path) if img.endswith(".jpg")]

        random.shuffle(images)
        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for img in train_images:
            src_img_path = os.path.join(label_path, img)
            dest_img_path = os.path.join(train_path, label_dir, img)
            shutil.copy(src_img_path, dest_img_path)
            print(f"Copied {src_img_path} to {dest_img_path}")

        for img in val_images:
            src_img_path = os.path.join(label_path, img)
            dest_img_path = os.path.join(val_path, label_dir, img)
            shutil.copy(src_img_path, dest_img_path)
            print(f"Copied {src_img_path} to {dest_img_path}")

        for img in test_images:
            src_img_path = os.path.join(label_path, img)
            dest_img_path = os.path.join(test_path, label_dir, img)
            shutil.copy(src_img_path, dest_img_path)
            print(f"Copied {src_img_path} to {dest_img_path}")

print("Train, validation, and test directories created successfully!")


####################################################
#       Create Train, Valid and Test sets
####################################################
train_data_path = 'images\\train' 
test_data_path = 'images\\test'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
for data_path in glob.glob(train_data_path + '\\*'):
    classes.append(data_path.split('\\')[-1]) 
    train_image_paths.append(glob.glob(data_path + '\\*'))
    
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

#2.
# split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

#3.
# create the test_image_paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '\\*'):
    test_image_paths.append(glob.glob(data_path + '\\*'))

test_image_paths = list(flatten(test_image_paths))

print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

#######################################################
#      Create dictionary for class indexes
#######################################################

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

class SkinMoleDataset(Dataset):
    def __init__(self,image_paths,transform = False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('\\')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    
train_dataset = SkinMoleDataset(train_image_paths,train_transforms)
valid_dataset = SkinMoleDataset(valid_image_paths,test_transforms)
test_dataset = SkinMoleDataset(test_image_paths,test_transforms)

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])

#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################

def visualize_augmentations(dataset, idx=0, samples=10, cols=2, random_img = False):
    
    dataset = copy.deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout()
    plt.show()   

visualize_augmentations(train_dataset,np.random.randint(1,len(train_image_paths)), random_img = True)




train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# !!!!!! __all__ is defined as a list containing the names of the objects you want to be public !!!!!!
__all__ = ['train_loader', 'valid_loader', 'test_loader', 'idx_to_class', 'class_to_idx']

print("All the steps for dataset preparation is completed successfuly, Now it can be used in the main function!!")