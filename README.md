# Road Sign Detection using Keras and OpenCv

## Team Members involved:

#### Prajwal Venkatesh 
#### Rahul K R
#### Sachin S
#### Sambit Sanyal

![image.PNG](attachment:image.PNG)

### You can get the dataset from this link 

https://drive.google.com/drive/folders/1VtcyQnf6HgbQzlCQeSrNbn4pWiMwjlew?usp=sharing 

##### signnames.csv – It has all the labels and their descriptors.
##### train.p – It contains all the training image pixel intensities along with the labels.
##### valid.p – It contains all the validation image pixel intensities along with the labels.
##### test.p – It contains all the testing image pixel intensities along with the labels.
The above files with extension .p are called pickle files, which are used to serialize objects into character streams. These can be deserialized and reused later by loading them using the pickle library in python.

Let’s implement a Convolutional Neural Network (CNN) using Keras in simple and easy-to-follow steps. A CNN consists of a series of Convolutional and Pooling layers in the Neural Network which map with the input to extract features. A Convolution layer will have many filters that are mainly used to detect the low-level features such as edges of a face. The Pooling layer does dimensionality reduction to decrease computation. Moreover, it also extracts the dominant features by ignoring the side pixels. To read more about CNNs, go to this link.



### Importing the libraries
We will be needing the following libraries. Make sure you install NumPy, Pandas, Keras, Matplotlib and OpenCV before implementing the following code.


```python
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator 
import pickle 
import pandas as pd 
import random 
import cv2 

np.random.seed(0) 

```


```python
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    

#### Loading the dataset
Time to load the data. We will use pandas to load signnames.csv, and pickle to load the train, validation and test pickle files. After extraction of data, it is then split using the dictionary labels “features” and “labels”.


```python
# Read data 
data = pd.read_csv("signnames.csv") 
df = data['SignName'].values.tolist()
num_classes = len(df)

with open('train.p', 'rb') as f: 
    train_data = pickle.load(f) 
with open('valid.p', 'rb') as f: 
    val_data = pickle.load(f) 
with open('test.p', 'rb') as f: 
    test_data = pickle.load(f) 

# Extracting the labels from the dictionaries 
X_train, y_train = train_data['features'], train_data['labels'] 
X_val, y_val = val_data['features'], val_data['labels'] 
X_test, y_test = test_data['features'], test_data['labels'] 

#Storing the shapes
train_shape = X_train.shape
val_shape = X_val.shape
test_shape = X_test.shape

# Printing the shapes 
print(train_shape)
print(val_shape) 
print(test_shape) 

```

    (34799, 32, 32, 3)
    (4410, 32, 32, 3)
    (12630, 32, 32, 3)
    

#### Preprocessing the data using OpenCV
Preprocessing images before feeding into the model gives very accurate results as it helps in extracting the complex features of the image. OpenCV has some built-in functions like cvtColor() and equalizeHist() for this task. Follow the below steps for this task –

First, the images are converted to grayscale images for reducing computation using the cvtColor() function.
The equalizeHist() function increases the contrasts of the image by equalizing the intensities of the pixels by normalizing them with their nearby pixels.
At the end, we normalize the pixel values between 0 and 1 by dividing them by 255.


```python

def preprocessing(img): 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = cv2.equalizeHist(img) 
    img = img / 255
    return img 

X_train = np.array(list(map(preprocessing, X_train))) 
X_val = np.array(list(map(preprocessing, X_val))) 
X_test = np.array(list(map(preprocessing, X_test))) 

X_train = X_train.reshape(*train_shape[0:3],1) 
X_val = X_val.reshape(*val_shape[0:3], 1) 
X_test = X_test.reshape(*test_shape[0:3], 1) 

```

#### Post Loading the data
After reshaping the arrays, it’s time to feed them into the model for training. But to increase the accuracy of our CNN model, we will involve one more step of generating augmented images using the ImageDataGenerator.

This is done to reduce overfitting the training data as getting more varied data will result in a better model. The value 0.1 is interpreted as 10%, whereas 10 is the degree of rotation. We are also converting the labels to categorical values, as we normally do.


```python
datagen = ImageDataGenerator(width_shift_range = 0.1, 
                height_shift_range = 0.1, 
                zoom_range = 0.2, 
                shear_range = 0.1, 
                rotation_range = 10) 
datagen.fit(X_train) 

y_train = to_categorical(y_train, num_classes) 
y_val = to_categorical(y_val, num_classes) 
y_test = to_categorical(y_test, num_classes) 

```

#### Building the model
As we have 43 classes of images in the dataset, we are setting num_classes as 43. The model contains two Conv2D layers followed by one MaxPooling2D layer. This is done two times for the effective extraction of features, which is followed by the Dense layers. A dropout layer of 0.5 is added to avoid overfitting the data.


```python
def cnn_model(): 
    model = Sequential() 
    model.add(Conv2D(60, (5, 5), 
                    input_shape =(32, 32, 1), 
                    activation ='relu')) 

    model.add(Conv2D(60, (5, 5), activation ='relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(30, (3, 3), activation ='relu')) 
    model.add(Conv2D(30, (3, 3), activation ='relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Flatten()) 
    model.add(Dense(500, activation ='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(num_classes, activation ='softmax')) 
    
    # Compile model 
    model.compile(Adam(lr = 0.001), 
                loss ='categorical_crossentropy', 
                metrics =['accuracy']) 
    return model 

model = cnn_model() 
history = model.fit_generator(datagen.flow(X_train, y_train, 
                            batch_size = 50), steps_per_epoch = 600, 
                            epochs = 15, validation_data =(X_val, y_val), 
                            shuffle = 1) 

model.save('mymodel.hdf5')

with open('history.csv', 'w') as f: 
    pd.DataFrame(history.history).to_csv(f) 

```

    Epoch 1/15
    600/600 [==============================] - 7s 12ms/step - loss: 2.7665 - accuracy: 0.2480 - val_loss: 0.3545 - val_accuracy: 0.9000
    Epoch 2/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.7365 - accuracy: 0.7719 - val_loss: 0.1437 - val_accuracy: 0.9565
    Epoch 3/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.4019 - accuracy: 0.8736 - val_loss: 0.0810 - val_accuracy: 0.9739
    Epoch 4/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.3044 - accuracy: 0.9040 - val_loss: 0.0603 - val_accuracy: 0.9785
    Epoch 5/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.2492 - accuracy: 0.9231 - val_loss: 0.0775 - val_accuracy: 0.9746
    Epoch 6/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.2017 - accuracy: 0.9370 - val_loss: 0.0360 - val_accuracy: 0.9889
    Epoch 7/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.1705 - accuracy: 0.9466 - val_loss: 0.0354 - val_accuracy: 0.9893
    Epoch 8/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.1625 - accuracy: 0.9499 - val_loss: 0.0345 - val_accuracy: 0.9896
    Epoch 9/15
    600/600 [==============================] - 9s 15ms/step - loss: 0.1454 - accuracy: 0.9557 - val_loss: 0.0471 - val_accuracy: 0.9848
    Epoch 10/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.1295 - accuracy: 0.9587 - val_loss: 0.0426 - val_accuracy: 0.9880
    Epoch 11/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.1155 - accuracy: 0.9648 - val_loss: 0.0331 - val_accuracy: 0.9902
    Epoch 12/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.1209 - accuracy: 0.9634 - val_loss: 0.0321 - val_accuracy: 0.9905
    Epoch 13/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.0973 - accuracy: 0.9688 - val_loss: 0.0341 - val_accuracy: 0.9914
    Epoch 14/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.1070 - accuracy: 0.9667 - val_loss: 0.0393 - val_accuracy: 0.9921
    Epoch 15/15
    600/600 [==============================] - 7s 11ms/step - loss: 0.0950 - accuracy: 0.9709 - val_loss: 0.0429 - val_accuracy: 0.9889
    

#### Storing and loading the Model in mymodel.hdf5

We stored the model and also  the training history data as well


```python
loadedModel = keras.models.load_model('mymodel.hdf5')

loadedHistory = pd.read_csv('history.csv')
```

#### Plotting the Graph (A)

We will now a plot of graph of loss with the number of Epoch


```python
plt.plot(loadedHistory['loss']) 
plt.plot(loadedHistory['val_loss']) 
plt.legend(['training', 'validation']) 
plt.title('Loss') 
plt.xlabel('epoch') 
```




    Text(0.5, 0, 'epoch')




![svg](output_17_1.svg)


#### Plotting the Graph (B)

We will now a plot of graph of Accuracy with the number of Epoch


```python
plt.plot(loadedHistory['accuracy']) 
plt.plot(loadedHistory['val_accuracy']) 
plt.legend(['training', 'validation']) 
plt.title('Accuracy') 
plt.xlabel('epoch') 
```




    Text(0.5, 0, 'epoch')




![svg](output_19_1.svg)


### Now we verify the Test loss and Accuracy
Here we have it


```python
score = loadedModel.evaluate(X_test, y_test, verbose = 0) 
print('Test Loss: ', score[0]) 
print('Test Accuracy: ', score[1]) 

```

    Test Loss:  0.1362009197473526
    Test Accuracy:  0.9662708044052124
    

### We now test it some testing data


```python
x=990


plt.imshow(X_test[x].reshape(32, 32)) 
print("Predicted sign: "+ df[int( 
            loadedModel.predict_classes(X_test[x].reshape(1, 32, 32, 1)))])
    

```

    Predicted sign: Speed limit (20km/h)
    


![svg](output_23_1.svg)



```python
 x=100

plt.imshow(X_test[x].reshape(32, 32)) 
print("Predicted sign: "+ df[int( 
            loadedModel.predict_classes(X_test[x].reshape(1, 32, 32, 1)))])

```

    Predicted sign: Speed limit (30km/h)
    


![svg](output_24_1.svg)



```python
x=150
plt.imshow(X_test[x].reshape(32, 32)) 
print("Predicted sign: "+ df[int( 
            loadedModel.predict_classes(X_test[x].reshape(1, 32, 32, 1)))])
```

    Predicted sign: Speed limit (100km/h)
    


![svg](output_25_1.svg)



```python
x=275
plt.imshow(X_test[x].reshape(32, 32)) 
print("Predicted sign: "+ df[int( 
            loadedModel.predict_classes(X_test[x].reshape(1, 32, 32, 1)))])
```

    Predicted sign: General caution
    


![svg](output_26_1.svg)



```python
x=420
plt.imshow(X_test[x].reshape(32, 32)) 
print("Predicted sign: "+ df[int( 
            loadedModel.predict_classes(X_test[x].reshape(1, 32, 32, 1)))])
```

    Predicted sign: Speed limit (60km/h)
    


![svg](output_27_1.svg)

