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


To add a new dataset, just write a new method readAndPrintDataImagesXXX