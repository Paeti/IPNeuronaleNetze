### The imdb_script.py will

1. download **imdb_crop.tar** if it doesn't exist.
2. read out the image information from the **imdb_metadata.csv** file.
3. assign the information to the images in array
4. split the data into **training-, validation-, testset**
5. write the datasets into **tf_records**


#### in Detail
1. Script
   * checks if **imdb_crop.tar** already exists in *../data/*
   * if not: downloads it from internet
   * checks if **imdb_crop** tarfile is already unpacked
   * if not: unpacks it to *../data/imdb_crop*
2. Script
   * reads out **imdb_metadata.csv** (it needs to be in the same folder) 
   * saves date of birth of person who is on image, path where image lies in imdb_crop file, gender of person on image, time when photo was taken, facelocation on the image
3. Script
   * saves date of birth (if it is valid), gender, face_location suitable to every imagepath in array I 
   * calculates the age of the person with the information about date of birth and when the photo was taken
   * if the date of birth is not valid, the age is set to -1
4. Script
   * creates array **X** for the images 
   * creates array **Y_age** for the appropriate information about the age of the person on the image at the same position in X
   * creates array **Y_gender** for the appropriate information about the gender
   * You can **set the number of images** for the training (here 75% of all saved images in I) and validationset (here 20% of all saved images in I) yourself with size_training and size_val. The used values are recommended
5. Method: **write_tfrecord(datasetX, datasetY, t)**
   * writes image and the information about it info tfrecord file so tensorflow can work with it
   Script
   * creates tfrecord files for age and gender sets for each dataset