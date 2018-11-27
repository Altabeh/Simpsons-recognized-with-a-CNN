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
the label that counts as the name of the folder it is being fed from.

```ruby
              R = im[:, :, 0]
              G = im[:, :, 1]
              B = im[:, :, 2]
              x_test.append([R, G, B])  
              y_test.append([label]) 
```
We then convert integer labels to one-hot vectors of size (1,4)
using keras.utils for use with categorical_crossentropy.

```ruby
y_train_cat = keras.utils.to_categorical(y_train, num_class) 
y_test_cat = keras.utils.to_categorical(y_test, num_class)
y_val_cat = keras.utils.to_categorical(y_val, num_class) 
```

![alt text](https://github.com/Altabeh/Simpsons-recognized-with-a-CNN/blob/master/simpson-family.gif)
