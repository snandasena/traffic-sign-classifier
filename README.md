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

###### Discussions: Data preprocessing

* Grayscaled RGB images to reduce three color channels to one channel. This will reduce CNN training time and it'll help to identify edges clearly
* Normalized image data into range [-1, 1]. If we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ from one another. We might be over compensating a correction in one weight dimension while undercompensating in another

#### Image Data Augmentation
Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset. Following image processing techniques will be used to augmentations for images. Following common augmentation techinues will be used to do image data augmentation.

Following augmentation techniques were used to do image data augmentation.

###### Translation

```python
def random_translate(img):
    """
    This is used to apply linear transfomation followed by vector addition(translation). Also this technique
    is called as Affine transform.
    https://docs.opencv.org/4.4.0/d4/d61/tutorial_warp_affine.html
    
    :param img - grayscale and normalized image
    """
    h,w = img.shape[:2]
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(w,h))
    
    return dst[:,:,np.newaxis]
    
```

###### Scaling

```python
def random_scaling(img):   
    """
    This is used to gerate new scaled images.
    
    :param img - grayscale and normalized image
    
    """
    h,w = img.shape[:2]

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[h-px,px],[px,w-px],[h-px,w-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[h,0],[0,w],[h,w]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(h,w))
    
    return dst[:,:,np.newaxis]
    
```

###### Warping

```python
def random_warp(img):
    """
    This is ued to do warp images and affine transformation technique is used to generate image mstrix.
    
    :param img - grayscale and normalized image
    """
    h,w = img.shape[:2]
    # random scaling coefficients
    dx = np.random.rand(3) - 0.5
    dx *= w * 0.06   # this coefficient determines the degree of warping
    dy = np.random.rand(3) - 0.5
    dy *= h * 0.06
    # 3 starting points for transform, 1/4 way from edges
    x1 = w/4
    x2 = 3*w/4
    y1 = h/4
    y2 = 3*h/4
    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+dy[0],x1+dx[0]],
                       [y2+dy[1],x1+dx[1]],
                       [y1+dy[2],x2+dx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(w,h))
    
    return dst[:,:,np.newaxis]
```

###### Brightening 

```python
def random_brightness(img):
    """
    This is used to change brightness randomly.
    
    :param img - grayscale and normalized image
    """
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    
    return dst
    
```    
