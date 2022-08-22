# Leaf Disease Classification
## Introduction
For this classification task, a total of 8010 representative sample images were used to develop an automatic Faba Bean leaf disease classification system using a convolutional neural network (CNN). First, I have developed the segmentation algorithm based on Canny edge detector and removed the image background from the region of interest. Afterward, I have designed a new CNN network architecture and a modified version of AlexNet architecture to fit the classification task. In addition, ResNet50V2 and InceptionV3 pre-trained models were chosen due to their better performance in such tasks. The FBDCNet model (which I have designed it from scratch) is trained with both raw images and segmented images. When it is trained with raw images an overall testing accuracy of 90% under the given test set is achieved. Compared to AlexNet, InceptionV3, and ResNet50 the classification accuracy exceeds by 2%, 6%, and 1% respectively. In the same way training of FBDCNet model with segmented images give a 98% testing accuracy under the given test set. Compared to AlexNet, InceptionV3 and ResNet50, the classification accuracy increased by 5%, 2%, and 1% respectively. The experimental results demonstrate that the proposed Canny edge detection based segmentation technique improves the total testing accuracy of the FBDCNet model by 8%. Thus, it can be seen that the proposed model effectively identifies Faba Bean leaf diseases with the proposed segmentation technique. Meanwhile, this study explores a new approach for the accurate classification of leaf diseases that provides a theoretical foundation for the application of deep learning in the field of agricultural information.

## System Architecture
![alt text](![image](https://user-images.githubusercontent.com/20771402/185979962-f747cf34-8d37-4f62-a93b-b0017d7e2c82.png?raw=true)

## Data

The overall data used for training the proposed models was collected for by the author and can be found in this link.


## Installation

- **The Repository**
```
git clone https://github.com/bkget/Leaf-Disease-Classification.git
cd Leaf-Disease-Classification
pip install -r requriements.txt
```

- **Jupyter Noteboks**
```
cd notebooks/with_segmented_images
jupyter notebook From_Scratch
```
