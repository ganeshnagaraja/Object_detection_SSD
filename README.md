This repository includes:

- Model training: Code to train the deep neural networks to predict bounding boxes and classes of objects
- Pre Processing Script: Scripts to process input data for dataloader

Contact:

If you have any questions/suggestions or find any bugs, please let me know: Ganesh Nagaraja - ganesh_nagaraja@outlook.com

## Installation

Ubuntu 16.04 and Python3 is used for the code

### Dependencies

pip install requirements.txt

## Setup

1. Download the datasets for training from https://github.com/gulvarol/grocerydataset and combine both the parts
   Replace the shelf_images folder with https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz

2. Download the model checkpoints from [Google Drive](https://drive.google.com/open?id=1cgKhXtFe0eUJrLMt8ptRactHGRzfgjob)

3. Download dependencies as seen at the last section of this readme.

### Note
Ther are Multiple(10) classes, but they have been combined into one class for the purpose of this objective( since the number of objects per class is low and it requires a lot of data augmentation and training)

## To run the code:

### 1. Training Code

The folder `Object_detection_SSD` contains the code used to train the
object detection model.

Data Preparation:

Training images:
- Images are present in the `ShelfImages` folder.
- Naming convention is as follows - SHELF_NAME_SNAPSHOT.JPG
- Labels --> Class: Since we only need to identify if the object is present or not, we will combine all the classes to one class.
  BBoxes: Folder `ProductImagesFromShelves` contains the respective class folders along with images and these images' names contain bounding boxes coordinates for individual object.
  Eg: C1_P01_N1_S3_1.JPG_1276_1828_276_448.png -- Traning image C1_P01_N1_S3_1.JPG contains an object whose
  cordinates are 1276_1828_276_448 (x-min, y-min, width, height)
- Scripts --> `create_data_lists.py` script first creates a 'file_name_with_details.csv' which contains information such as
  the orginial file name present in ShelfImages folder, object coordinates, class of the object, etc.
  Secondly the script uses this information to create a two JSON files - `TRAIN_images.json` containing all the images names along with the path and second file `TRAIN_objects.json` contains bounding boxes and class labels for the respective image.
- Augmentation -->
zoom out with 50% chance of occurance(randomness of 0.5) - to be able to detect smaller images.   
zoom in (random crop) - to detect larger and partial images.    
flip images horizontally.   
photometric distort - brightness, contrast, staturation and hue. Each with a 50% chance

To train:
- Edit the `config.yaml` file to fill in the paths to the dataset, select hyperparameter values, etc. All the parameters are explained in comments within the config file.
- Start training: `python3 train.py -c config.yaml`

Model:
- SSD300 model with backbone of VGG16 pretrained on Imagenet dataset.

To run Validation:
- Edit the `config.yaml` file to fill in the paths to the dataset, checkpoint etc. All the parameters are explained in comments within the config file.
- Run validation: `python3 eval.py -c config.yaml`
- Calculates the metric on the validation dataset such as mAP, precision, recall

Inferenc:
- Edit the `config.yaml` file with the path to the test images folder and chose the right checkpoint
- run `python3 inference.py -c config.yaml`