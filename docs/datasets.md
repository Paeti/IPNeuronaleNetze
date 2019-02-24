The "*datasets.py*" file is a class which implements useful methods
to process files from a given dataset.

* def createFolder(self, __directory__):
	creates a Folder at __directory__

* def createClassificationFolders(self, __directory__):
	creates Folders ranging from 0 to 100 in a given __directory__

* def downloadAndUnzip(self, __url__, __target_zip_directory__, __target_extract_directory__):
	downloads file from __url__ into __target_zip_directory__ and extracts
	files into target_extract_directory

* def readAndPrintDataImagesLAP(self, __image_directory__, __type_set_csv__, __classification_target_directory__):
	loads LAP dataset images from __image_directory__ and finds the corresponding age
	from the csv file located in __image_directory__, split into three __type_set_csv__ (train, valid, test).
	These images are written in the correct __classification_target_directory__

* def load_images_from_folder(self, __folder__):
    loads all images from __folder__

* def readAndPrintDataImagesFGNET(self, __image_directory__, classification_target_directory):
	loads  FGNET dataset images from __image_directory__ and finds the corresponding age from the image name.
	These images are randomly written into either Train, Test, Valid Classification folder and further into the folders of the corresponding age

* def readAndPrintDataImagesIMDB(self, csv_directory, image_directory, classification_folder):
	loads IMDB dataset images from __image_directory__ and finds the corresponding age and gender values from the csv file at __csv_directory__

* def print_images(self, datasetX, datasetY, t, classification_folder):
	called by __readAndPrintDataImagesIMDB__ and uses that __classification_folder__ where __t__ is the age/gender folder with corresponding Train/Valid/Test folders. __datasetX__ are the images and __datasetY__ the labels



Example

```
dataset = Dataset()
dataset.createFolder("../data/classification/age")
dataset.createClassificationFolders("../data/classification/age/Train")
dataset.createClassificationFolders("../data/classification/age/Valid")
dataset.createClassificationFolders("../data/classification/age/Test")
#LAP
dataset.downloadAndUnzip("http://158.109.8.102/AppaRealAge/appa-real-release.zip", "../data/appa-real-release.zip", "../data/appa-real-release")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "train", "../data/classification/age/Train")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "valid", "../data/classification/age/Valid")
dataset.readAndPrintDataImagesLAP("../data/appa-real-release", "test", "../data/classification/age/Test")

#FGNET
dataset.downloadAndUnzip("http://yanweifu.github.io/FG_NET_data/FGNET.zip", "../data/FGNET.zip", "../data/FGNET")
dataset.readAndPrintDataImagesFGNET("../data/FGNET", "../data/classification")
#IMDB
dataset.downloadAndUnpack("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb"
                         "_crop.tar", "../data/imdb_crop.tar", "../data/imdb_crop");
dataset.readAndPrintDataImagesIMDB('imdb_metadata.csv', "../data/imdb_crop", "../data/classification")
```

To add a new dataset, just write a new method readAndPrintDataImagesXXX