# Identifying Simpson family members by training a deep Convolutional Neural Network (CNN)
First things first, we'd like to extend our appologies to Santa's Little Helper, Marge and Snowball II for not being
included in our classifier.   

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

we have included a block in main_train.py that does this automatically:

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

![alt text](https://github.com/Altabeh/Simpsons-recognized-with-a-CNN/blob/master/simpson-family.gif)
