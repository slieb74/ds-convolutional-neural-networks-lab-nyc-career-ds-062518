
# Convolutional Neural Networks Lab

## Objective

In this lab, we'll learn and implement best practices for structuring a Deep Learning/Computer Vision project.  We'll then use design and train a Convolutional Neural Network to classify if an image contains Santa!

## 1. Properly store your images

When you're analyzing your image data, file management is important. We will be using the santa images again, but this time, they are just stored in two folders: `santa` and `not_santa`, under. We want to work with a `train`, `validation` and `test` data set now, as we know by now that this is the best way to go. 

Let's import libraries `os` and `shutil`, as we'll need them to create the new folders and move the new files in there.  Import these modules in the cell below. 

Create three objects representing the existing directories 'data/santa/' as `data_santa_dir` and 'data/not_santa/' as `data_not_santa_dir`. We will create a new directory 'split/' as `new_dir`, where we will split the data set in three groups (or three subdirectories) 'train', 'test' and 'validation', each containing `santa` and `not_santa` subfolders. The final desired structure is represented below:

![title](folder_structure.png)


```python
data_santa_dir = 'data/santa/'
data_not_santa_dir = 'data/not_santa/'
new_dir = 'split/'
```

You can use `os.listdir()` to create an object that stores all the relevant image names. 

Run the cell below to have create a list comprehension that reads and stores all filenames that end with `.jpg` inside of the `data_santa_dir` directory we stored earlier.


```python
imgs_santa = [file for file in os.listdir(data_santa_dir) if file.endswith('.jpg')]
```

Let's inspect the first 10 to see if it worked correctly.


```python
imgs_santa[0:10]
```

Let's see how many images there are in the 'santa' directory.  Complete the following print statement to determine how many total images are in this directory. 


```python
print('There are',len(None), 'santa images')
```

Now, repeat the process we did above, but for the `not_santa` directory.  Write the appropriate list comprehension below to create an array containing all the filenames contained within the `not_santa` directory.


```python
imgs_not_santa = None
```


```python
print('There are', len(None), 'images without santa')
```

Create all the folders and subfolder in order to get the structure represented above. You can use `os.path.join` to create strings that will be used later on to generate new directories.

In the cell below, create the `'split'` directory we'll need by calling `os.mkdir()` on the `new_dir` variable we created above. 

Now, run the cell below to join all the paths as needed.  Pay special attention to what we're doing at this step, and try to figure out what each path will contain.  If you're not sure, inspect them!


```python
train_folder = os.path.join(new_dir, 'train')
train_santa = os.path.join(train_folder, 'santa')
train_not_santa = os.path.join(train_folder, 'not_santa')

test_folder = os.path.join(new_dir, 'test')
test_santa = os.path.join(test_folder, 'santa')
test_not_santa = os.path.join(test_folder, 'not_santa')

val_folder = os.path.join(new_dir, 'validation')
val_santa = os.path.join(val_folder, 'santa')
val_not_santa = os.path.join(val_folder, 'not_santa')
```

Display the path for `train_santa` in the cell below.  Does this match with the path you thought it created?

Now use all the path strings you created to make new directories. You can use `os.mkdir()` to do this. Go have a look at your directory and see if this worked!

In the cell below, call `os.mkdir()` on every path we created in the cell above. 

Copy the Santa images in the three santa subfolders. Let's put the first 271 images in the training set, the next 100 images in the validation set and the final 90 images in the test set.

The code for moving the images into the training set has been provided for you.  Use this as a reference to complete the validation santa and test santa cells!


```python
# train santa
imgs = imgs_santa[:271]
for img in imgs:
    origin = os.path.join(data_santa_dir, img)
    destination = os.path.join(train_santa, img)
    shutil.copyfile(origin, destination)
```


```python
# validation santa
imgs = None

```


```python
# test santa
imgs = None

```

Now, repeat all this for the 'not_santa' images!


```python
# train not_santa
imgs = None 



# validation not_santa
imgs = None



# test not_santa
imgs = None



```

Let's print out how many images we have in each directory so we know for sure our numbers are right!  Complete each of the following print statements to examine how many images we have in each subdirectory.  


```python
print('There are', len(os.listdir(None)), 'santa images in the training set')
# Expected Output: There are 271 santa images in the training set
```


```python
print('There are', len(os.listdir(None)), 'santa images in the validation set')
# Expected Output: There are 100 santa images in the validation set
```


```python
print('There are', len(os.listdir(None)), 'santa images in the test set')
# Expected Output: There are 90 santa images in the test set
```


```python
print('There are', len(os.listdir(None)), 'images without santa in the train set')
# Expected Output: There are 271 images without santa in the train set
```


```python
print('There are', len(os.listdir(None)), 'images without santa in the validation set')
# Expected Output: There are 100 images without santa in the validation set
```


```python
print('There are', len(os.listdir(None)), 'images without santa in the test set')
# Expected Output: There are 90 images without santa in the test set
```

## Data Preprocessing

We'll make use of the image preprocessing functionality found in keras to reshape our data as needed.  

Now that we've sorted our data, we can easily use Keras' module with image-processing tools. Let's import the necessary libraries below. Run the cell below to import everything we'll need. 


```python
import time
import matplotlib.pyplot as plt
import scipy
import numpy as np
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

np.random.seed(123)
```

Now, complete the image generator code in the cell below.  The code for test generator has been completed for you--use this as a reference to complete the `val_generator` and `train_generator` objects.  Keep the parameters the same across all 3 of them, so that there won't be any discrepancies between how the images look in each directory. 

(**_NOTE:_** The directory path referenced will depend on which set it is targeting, and the `batch_size` parameter will correspond to the number of images in the corresponding set, so these parameters will depend on if it referencing test, val, or train!)


```python
# get all the data in the directory split/test (180 images), and reshape them
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_folder, 
        target_size=(64, 64), batch_size = 180) 

# get all the data in the directory split/validation (200 images), and reshape them
val_generator = None

# get all the data in the directory split/train (542 images), and reshape them
train_generator = None
```

Now, in the cell below, separate out the images and labels by calling `next()` on the appropriate generators.  The first one has been completed for you.  


```python
# create the data sets
train_images, train_labels = next(train_generator)
test_images, test_labels = None
val_images, val_labels = None
```

Now, run the cell below to get summary statistics on the current shape of our data.


```python
# Explore your dataset again
m_train = train_images.shape[0]
num_px = train_images.shape[1]
m_test = test_images.shape[0]
m_val = val_images.shape[0]

print ("Number of training samples: " + str(m_train))
print ("Number of testing samples: " + str(m_test))
print ("Number of validation samples: " + str(m_val))
print ("train_images shape: " + str(train_images.shape))
print ("train_labels shape: " + str(train_labels.shape))
print ("test_images shape: " + str(test_images.shape))
print ("test_labels shape: " + str(test_labels.shape))
print ("val_images shape: " + str(val_images.shape))
print ("val_labels shape: " + str(val_labels.shape))
```

If these numbers look confusing, don't sweat it--these are **_Tensors_**.  A tensor is an n-dimensional array.  Take a look at the the shape of `train_images`:  
<br>  
<center>`(542, 64, 64, 3)`</center>

In plain English, we would read this as "`train_images` contains 542 images.  These images are 64 pixels wide, 64 pixels high, and have 3 color channels, meaning that these are color images".  

Since we'll be using a Multi-Layer Perceptron as our baseline, that means we'll need to create a version of our data that is reshaped from tensors to vectors. 

In the cell below, complete the code to reshape our images.  The first one has been provided for you. 


```python
train_img = train_images.reshape(train_images.shape[0], -1)
test_img = None
val_img = None

print(train_img.shape) # Expected Output: (542, 12288)
print(test_img.shape) # Expected Output: (180, 12288)
print(val_img.shape) # Expected Output: (200, 12288)
```

We also need to rehape our labels accordingly.  In the cell below, complete the code to reshape our labels. The first one has been provided for you.  

**_Hint:_** Pay attention to the dimensionality in the 2nd parameter inside the `.reshape` call--it should align with the number of items contained in that set!


```python
train_y = np.reshape(train_labels[:,0], (542,1))
test_y = None
val_y = None
```

### Building our Baseline MLP

Since we want to see what kind of performance gains a CNN gives us a basic MLP, that means we'll need to build and train a basic MLP first!

In the cell below, build and train an MLP with the following specifications:

* Import whatever you'll need to build the network from keras
* `input_shape=(12288,)` (remember to declare this when creating the first hidden layer)
* Hidden Layer 1: 20 neurons, relu activation
* Hidden Layer 2: 7 neurons, relu activation
* Hideen Layer 3: 5 neurons, relu activation
* Output layer: 1 neuron, sigmoid activation

Now, compile the model with the following hyperparameters:

* `'adam'` optimizer
* `'binary_crossentropy'` for loss
* set metrics to `['accuracy']`

Now, fit the model.  In addition to passing in our training data and labels, also set epochs to `50`, batch size to `32`, and pass in our validation data as well. 


```python
mlp_history = None
```

Now, get our final training and testing results using `model.evaluate` and passing in the appropriate set of data/labels in the cells below. 


```python
results_train = None
```


```python
results_test = None
```


```python
results_train
```


```python
results_test
```

Finally, let's plot our accuracy and our loss.


```python
history = mlp_history.history

plt.figure()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Accuracy/Validation Accuracy for MLP')
plt.xlabel('Epoch')
plt.legend(['accuracy', 'Val Accuracy'])
plt.show()
```

Our model is currently "thrashing"--it has not yet converged at the right values, and is extremely sensitive to noise. 

Remember that, in our lab on "building deeper neural networks from scratch, we got to a train set prediction was 95%, and a test set prediction of 74.23%.  

This result is similar to what we got building our manual "deeper" dense model. The results are not entirely different. This is not a surprise! We should note that there were some differences in our manual approach and int then model we just built:
- Before, we only had a training and a validation set (which was at the same time the test set). Now we have split up the data 3-ways.
- We didn't use minibatches before, yet we used mini-batches of 32 units here.


# 3. Convnet

Now, we'll build a Convolutional Neural Network to see how it measures up. 

In the cell below, create a CNN with the following specifications:

* an `input_shape` of `(64, 64, 3)` (still declare this during the creating of the first hidden layer)
* Layer 1: Conv2D, 32 filters, filter size `(3,3)`, `relu` activation
* Layer 2: MaxPooling2D, shape `(2,2)`
* Layer 3:  Conv2D, 32 filters, filter size `(4, 4)`, `relu` activation
* Layer 4: MaxPooling2D, shape `(2,2)`
* Layer 5: Conv2D, 64 filters, filter size `(3,3)`, `relu` activation
* Layer 6: a `Flatten()` layer
* Layer 7: Dense layer, 64 neurons, `relu` activation
* Layer 8: Output layer, 1 neuron, `sigmoid` activation


```python
model = None


```

Now, compile the model with the same parameters we used for our MLP above in the cell below.

Finally, fit the model.  In addition to our training images and labels, set epochs to `30`, batch size to `32`, and also pass in our validation data.


```python
cnn_history = None
```

Now, let's get the final results for our training and testing sets by calling `model.evaluate` and passing in the appropriate sets. 


```python
results_train = None
```


```python
results_test = None
```


```python
results_train
```


```python
results_test
```

Let's plot the accuracy of our CNN results in the cell below, as we did for our MLP.


```python
history = cnn_history.history

plt.figure()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Accuracy/Validation Accuracy for CNN')
plt.xlabel('Epoch')
plt.legend(['accuracy', 'Val Accuracy'])
plt.show()
```

#### Interpreting Our Results

Our model is still thrashing a bit, but the overall performance is much higher.  More importantly, it doesn't seem as if the model has overfit, whereas there was definitely evidence of overfitting when looking at the results from our MLP. 

To end this lab, we'll get some practice with Data Augmentation. This is a very useful trick for generating more data, while also stopping the model from overfitting on certain quirks found in the images!

# Data augmentation

ImageDataGenerator becomes really useful when we *actually* want to generate more data. We'll show you how this works.

This generator takes in a range of parameters we can set to tell it exactly how we want to to modify the images it receives.  The original image will be left alone, and copies will be made according to these specifications, resulting in images that are slightly different, which we can add to our training and testing sets. 

Run the cell below to create a data generator.  


```python
train_datagen= ImageDataGenerator(rescale=1./255, 
     rotation_range=40,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.3,
     zoom_range=0.1,
     horizontal_flip = True)
```

Now, we'll need to actually use the generator to generated augmented data. Run the cell below.


```python
names = [os.path.join(train_santa, name) for name in os.listdir(train_santa)]
img_path = names[91]
img = load_img(img_path, target_size=(64, 64))

reshape_img = img_to_array(img) 
reshape_img = reshape_img.reshape((1,) + reshape_img.shape) 
i=0
for batch in train_datagen.flow(reshape_img, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i % 3 == 0:
        break
plt.show()
```

Finally, we'll need to rescale everything as we did in the original data. Run the cell below to do this.  


```python
# get all the data in the directory split/test (180 images), and reshape them
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_folder, 
        target_size=(64, 64), 
        batch_size = 180,
        class_mode='binary') 

# get all the data in the directory split/validation (200 images), and reshape them
val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_folder, 
        target_size=(64, 64),
        batch_size = 32,
        class_mode='binary')

# get all the data in the directory split/train (542 images), and reshape them
train_generator = train_datagen.flow_from_directory(
        train_folder, 
        target_size=(64, 64), 
        batch_size = 32, 
        class_mode='binary')
```

Now, in the cell below, recreate the CNN model we did above.  Compile the model with the same specifications.  


```python
model = None
```

Finally, we'll need to fit the model.  This time, instead of passing in an array of data, we'll pass in the `train_generator` object we created! 

In addition to passing in `train_generator`, pass in the following parameters:

* `steps_per_epoch=25`
* `epochs=30`
* `validation_data=val_generator`
* `validation_steps=25`


```python
history_2 = None
```

Now, run the cell below to create a test set and labels we can use for checking the performance with `model.evaluate()`


```python
test_x, test_y = next(test_generator)
```

Now, run call `model.evaluate()` on the test set we just created, and then examine to see our testing loss and testing accuracy.  


```python
results_test = None
```


```python
results_test
```

Finally, let's visualize the results.  


```python
history = history_2.history

plt.figure()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Accuracy/Validation Accuracy for CNN with Data Augmentation') 
plt.xlabel('Epoch')
plt.legend(['accuracy', 'Val Accuracy'])
plt.show()
```

Interesting results! Our model still seems to be thrashing a bit, but a little less so than before.  It's also worth noting that model performance took a dive in the last epoch or 2, but before the dive, was generally about as well as it was without data augmentation.  
