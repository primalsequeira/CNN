### Introduction

A **convolutional neural network (CNN) is a specific type of artificial neural network for supervised learning in order to analyse the data. CNNs are applied to image processing, natural language processing and other kind of tasks.**

Here, an atomic element is taken from the input image and it is shifted to every location in the image. This process is called Convolution. The convolution is demonstrated by shifting the filter to every location in the image. These filters are learned entirely based on the data.

![cnn](https://user-images.githubusercontent.com/67816058/87246609-18cfb900-c46c-11ea-90fb-cb9be83f3a1c.png)

# MNIST digit classification



In this example, we are classifying the digits using CNN. The dataset we will be using is MNIST. It consists of **`60,000`** small square 28x28 pixel greyscale images of handwritten single digits from 0 to 9 for training and **10,000** testing images.


![mnistdataset](https://user-images.githubusercontent.com/67816058/87246630-3e5cc280-c46c-11ea-8748-cd6c27fcb371.png)


MNIST dataset has a lot of handwritten images. So lets just look into any 5 random images of each of the digit.
Here's the code:
```markdown
import random
num_of_samples = []

cols = 5  # We will select 5 random images 
num_of_classes = 10 #each digit total: 10

fig, axs = plt.subplots(nrows=num_of_classes, ncols=cols,
                       figsize=(5, 10))
fig.tight_layout()
for i in range(cols):
  for j in range(num_of_classes):
    x_selected = training_images[training_labels == j]
    axs[j][i].imshow(x_selected[random.randint(0, len(x_selected -1)),
                                :, :],
                    cmap=plt.get_cmap('gray')) 
    axs[j][i].axis("off")
    if i==2:
      axs[j][i].set_title(str(j))
      num_of_samples.append(len(x_selected))
      
```

We can also see the number of images of each digit using the code below.
```markdown

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_of_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of images")
 ```
It’s a bar graph having "Class number" as X-axis and "Number of images" as Y-axis.

## Import all the required libraries
```markdown
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

```
## Splitting the data into training and testing data

Here we call load_data function on the mnist object and then split the entire dataset into two lists i.e. the `training values` and the `testing values`.
We do this splitting because we want to train our model with a large data(training_images) and test it with the data which the model had'nt seen previously(test_images).

```markdown

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

```
#### Let's check the size of training images and testing images.
```markdown
print(training_images.shape)
print(test_images.shape)
```
The first parameter gives the number of images we have. The 2nd and 3rd parameter are the pixel values of the images from x to y (28x28). This is because when we train our model, the image needs to be in 28x28 format.

#### Now Let's just print the first training image and its label.

```markdown
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
```
We notice here that the values are between 0 to 255. But it becomes easier to train the model, if the values are between 0 and 1.

# Preprocess the Data

Previously we have seen that the dimension of training data is (60000, 28, 28). But the model which we are using i.e CNN (Convolution Neural Network) requires one more dimension which states if the image is greyscale or not. If the image is in rgb we must reshape it to the greyscale format.

```markdown
training_images = training_images.reshape(training_images.shape[0], 28, 28,1)
test_images = test_images.reshape(test_images.shape[0], 28, 28,1)
```
Now check the dimension.

```markdown
print(training_images.shape)
print(test_images.shape)
```

Next, we apply **`to_categorical() function`** since we have multiclass results (10 output).
It converts a class vector (integers) to binary class matrix.

```markdown
# Example for to_categorical()
# z = [1, 2, 0, 3]
# tf.keras.utils.to_categorical(z, num_classes=4)

num_classes = 10
training_labels = keras.utils.to_categorical(training_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
training_images = training_images.astype('float32')
test_images = test_images.astype('float32')
```
Here it returns binary matrix representation of the input.

`Normalization` is done next by dividing training_images by 255.
```markdown
training_images  = training_images / 255.0
test_images = test_images / 255.0
```

# Create the model
We will be using the most straightforward API which is Keras. Therefore, we import the Sequential Model from Keras and add Conv2D, MaxPooling, Flatten, Dropout, and Dense layers.
```markdown
batch_size = 128
num_classes = 10
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
```
`Sequential()` define the SEQUENCE of layers in the neural network.
In the first convolution layer we are generating 64 filters. These filtes are 3x3, Their activation is `relu` where the negative values will be thrown away and the input_shape is as before (28, 28, 1). 1 indicates that we are using single byte for color depth as our images are greyscaled.


It will then create a Pooling layer. Its `MaxPooling` because we are taking the maximum value. Its a 2x2 pool which means for every 4 pixels the biggest one will be considered.


`Dropout()` is a regularization technique that is used to prevent the neural networks from overfitting. It randomly drops neurons along with their connections from the neural network during training in each iteration thereby preventing overfitting. 

`model.summary()` will allow us to inspect the layers of the model and see the journey of the image through the convolutions.

```markdown
model.summary()
```
Look into the output Shape. It might look like a bug because the input shape is 28 x 28.The key to this is that the filter is 3x3 filter.
When we start scanning the image from the top-left, we cant calculate the filter for the top-left because it doesnt have neighbours above it or to its left.



The next pixel to the right will not work either because it doesnt have neighbour above it. So the first pixel we can calculate is shown below.

![pooling](https://user-images.githubusercontent.com/67816058/87246648-60564500-c46c-11ea-81dc-a3e2a48844af.png)

![pooling1](https://user-images.githubusercontent.com/67816058/87246669-8380f480-c46c-11ea-9783-69549570764f.png)


That pixel has got all 8 neighbours.
This means we cant use one pixel all around the image.Hence the output will be 2 pixel smaller on X and 2 pixel smaller on Y. Hence its (26, 26).

Next is the first Maxpooling layer with pool_size of (2, 2) thus it turns every 4 pixels to 1. So our output reduces from (26, 26) to (13, 13).


The next convolution and Maxpooling layers will operate similar as before.


`Flatten` takes the square (28, 28) images and turns into one dimensional array.
So when we flatten the final output shape (5, 5, 64) we get 1600 elements.

`The Dense layer` here has 128 neurons and each layer of neurons needs an activation function which tells what to do. `Relu` is an activation function which removes off the negative values. `"If X>0 return X, else return 0"`

The last dense layer has 10 neurons because we have 10 classes of numbers in the dataset and also activation function used is `softmax` which takes the set of values and selects the largest one.

For eg: If the output of last layer is  [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], then it returns  [0, 0, 0, 0, 1, 0, 0, 0, 0].

# Compile and train the model

Next step is to compile the model with an optimizer and loss function.
The epochs equals 5 value means that it will go through the training loop 5 times. Make a guess, measure how good or how bad the guesses with the loss function, 
then use the optimizer and the data to make another guess and repeat this.

We then train the model using `model.fit()` which then fits your training data to your training labels.
```markdown
history = model.fit(training_images, training_labels,batch_size=batch_size,
                 epochs = 5, verbose = 1, validation_data = (test_images, test_labels))
```
Now as we are done with training, we can see the accuracy at the end of the final epoch which is 0.9831. This means that the neural network is about 98% accurate in classifying the training data.

# Evaluate the Model 
Evaluation is done on the test data. It returned 98% accuracy.

```markdown
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
# Prediction
```markdown
classifications = model.predict(test_images)
print(classifications[0])
```
```markdown
print(test_labels[0])
```

It creates a set of classifications for each of the test images, and then prints the first entry in the classifications. The output is a list of numbers. 

These numbers are probability that the item is each of the 10 classes. First value in the list is the probability that the image is a ‘0’, the next is a ‘1’ and so on.


The probability that the first image is a ‘7’ is the highest among all. So the neural network is telling that the first testing image is a ‘7’.

