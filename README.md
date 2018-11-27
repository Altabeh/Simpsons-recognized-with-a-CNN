# Identifying Simpson family members by training a deep Convolutional Neural Network (CNN)
First things first, we'd like to extend our appologies to Santa's Little Helper, Marge and Snowball II for not being
included in our classifier.   

## Disclaimer
Given the RGB nature of this exercise, it would take about an hour to finish the training part over 3000 sample images in 30 epochs using an average GPU even though the dataset size may not seem big. So it is strongly recommedned not to try this on a CPU computer unless the user does not have access to a GPU.


## Dataset 

The credit of collecting and preparing the Simpsons characters goes to <a href="https://www.kaggle.com/alexattia/the-simpsons-characters-datasetdataset" title="alexattia">alexattia</a> from Kaggle.
We are going to use a more narrowed down version of this dataset which would be enough for our purposes in this repository <a name="*">1</a>.
To see the original dataset please follow the Kaggle link above.

<sup>[1]We would like to thank <a href="https://appliedai.wordpress.ncsu.edu/" title="Behnam Kia">Behnam Kia</a> for putting this dataset together.</sup>

Please download the dataset from  <a href="https://drive.google.com/file/d/1jDQcJvCmPn7q-eo-cT2fioNY-spavmUH/view" title="this">this</a> link.

## Unzipping 

To unzip the main file

```ruby
the-simpsons-dataset.zip
```

after having it downloaded, you may use the following block in main_train.py that does this automatically:

```ruby
def extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        zip = zipfile.ZipFile(filename)
        sys.stdout.flush()
        zip.extractall(mainpath)
        zip.close()
        print("Extraction done!")
    return root

data_folder = extract(dest_filename)
```
We have three folders: train (3000 images), validation (1000 images) and test (800 images), each of which
contains four directories for each character. 
Once an image is read, the code takes the color data and stores it in an array together with 
the label that counts as the name of the folder it is being fed from. For example

```ruby
           R = im[:, :, 0]
           G = im[:, :, 1]
           B = im[:, :, 2]
           x_test.append([R, G, B])  
           y_test.append([label]) 
```
After assigning proper float types to each one of train, validation and test datasets, we convert integer labels to one-hot vectors of size (1,4)
using keras.utils for use with categorical_crossentropy:

```ruby
y_train_cat = keras.utils.to_categorical(y_train, num_class) 
y_test_cat = keras.utils.to_categorical(y_test, num_class)
y_val_cat = keras.utils.to_categorical(y_val, num_class) 
```

where num_class is set to 4.

## Data augmentation
A crucial step in training any CNN is to make sure that it avoids memorization of patterns seen
in the images for a given class. For instance, if Homer always looks something like
![alt text](https://github.com/Altabeh/Simpsons-recognized-with-a-CNN/blob/master/homer.jpg)
in his pictures, the neural network will take it for granted that the unseen images pretty much
come in the same shape. To surprise the network, we should augment data, which essentially amounts to generating
synthetic data from existing ones to increase the learning capacity and reduce memorization possibility. 
We can illustrate what goes in this process by taking the middle image above and generate different variations of
it such as
![alt text](https://github.com/Altabeh/Simpsons-recognized-with-a-CNN/blob/master/homer_data.jpg)
Note that this process is done by keras' useful ImageDataGenerator() function.
```ruby
datagen = train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale= 1. / 225)    #Important: we don't need to augment the validation AND test sets
train_generator = datagen.flow_from_directory(data_folder + '/train',  batch_size=batch_size, target_size=(img_w, img_h),
                                              class_mode='categorical', shuffle = True)
val_generator = val_datagen.flow_from_directory(data_folder + '/validation', batch_size=batch_size, target_size=(img_w, img_h),
                                              class_mode='categorical', shuffle = True)
test_generator = val_datagen.flow_from_directory(data_folder + '/test', batch_size=batch_size, target_size=(img_w, img_h),
                                              class_mode='categorical', shuffle = True)
```
Keep in mind that when data augmentation is mentioned, it only refers to taking <b>train</b> data and augmenting it. So leave validation and test datasets alone. All we do here is to rescale them to bring the values encoded to something between
0 and 1 to have the spread of data points reduced for better training later on.
## CNN Model
Finally it is time to design a 'good' CNN for our training purpose. A good CNN is one that
does the job with all we have done so far up to an accuracy of ~ 94%. You might wonder why I pulled this magic number and not 
anything else. All you have to remember is that machine learning has a theory behind it, I'd personally like to call it (discrete) <b>Morse theory</b> in mathematics that has been sucessfull applied to many physics problems (A cost function is a Morse function, see <a href="https://en.wikipedia.org/wiki/Morse_theory" title="this">this</a> page for a definition). But practicaclly speaking, it is not quite Morse theory telling us what a good model is. It is all about <b>experimentation</b>. Well, yeah it is all experimentally verified with some nominal fluctuation due to noise and local machine performance, and etc. 

Any good model (as far as my experiments are considered 'legit'!) I stumbled upon tends to have more neurons in the finally hidden layers. Something like this would work up to an accuracy of 94.25% I talked about previously:
```ruby
#Model definition
model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=new_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(num_class, activation='softmax'))
model.summary()
```
A yet another technique for avoiding memorization is the use of some sort of regularization 
in machine learning. The lore is that you don't want to get a high accuracy without a large 
learning capacity. So don't forget about those dropout layers; sure you want a smart model! 
Again, nothing about our model is cut-and-dried. Try your own model and see in what ways you can improve this.

## Transfer Learning

![alt text](https://github.com/Altabeh/Simpsons-recognized-with-a-CNN/blob/master/simpson-family.gif)
