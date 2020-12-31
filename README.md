Deep Learning
---
## Project: Build a Traffic Sign Recognition Classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Dataset Summary & Exploration
This dataset is indluded German traffic signs. Following is the summary of the data set.

Number of training examples |34799|
----------------------------|------|
Number of testing examples |2630 |
Image data shape |(32, 32, 3)|
Number of classes |43|

### Exploratory visualization of the dataset

Following are the sample images for the dataset.

![](resources/all-data-rgb.png)

Following is the histogram for labels vs frequecy distribution

![](resources/provided-data-histogram.png )

**Min number of image per class:  180**    
**Max number of image per class:  2010**


### Design and Test a Model Architecture

##### Pre-process the dataset 

Grayscle and normalization techniques were used to pre-process image data for image processing pipline.  
For grayscalling following snipets was used.

```python
# Graysscale the images - train set
X_train_rgb = np.copy(X_train)
X_train_gray = np.sum(X_train / 3, axis=3, keepdims=True)

# Graysscale the images - test set
X_test_rgb = np.copy(X_test)
X_test_gray = np.sum(X_test / 3, axis=3, keepdims=True)

# Graysscale the images - valid set
X_valid_rgb = np.copy(X_valid)
X_valid_gray = np.sum(X_valid / 3, axis=3, keepdims=True)

```

For normalization following snipets were used.  

```python
# Normalize train values
X_train_normalize = (X_train - 128) / 128
# Normalize test values
X_test_normalize = (X_test - 128) / 128
# Normalize valid values
X_valid_normalize = (X_valid - 128) / 128

```
Following is a sample result for grayscaled images.

![](resources/all-data-gray.png)

###### Discussions 01: The submission describes the preprocessing techniques used and why these techniques were chosen.
* Grayscaled RGB images to reduce three color channels to one channel. This will reduce CNN training time and it'll help to identify edges clearly
* Normalized image data into range [-1, 1]. If we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ from one another. We might be over compensating a correction in one weight dimension while undercompensating in another

