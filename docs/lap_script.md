### The "*lap-script.py*" file will -

<pre><code>
1. look if a file exists at "*../data/appa-real-release.zip*" exists - do nothing; else - download dataset from
*"http://158.109.8.102/AppaRealAge/appa-real-release.zip*"

2. look if an extracted file at "*../data/appa-real-release*" exists - do nothing; else - extract downloaded dataset

3. create Classification folders in "*../data/LAP*" ranging from 0 to 100

4. open images and creates an array of tupels with the corresponding age value from a csv file

5. write the images into the fitting Classification folder

6. create a "*img_counts.txt*" file with the amount of Training, Validiation and Test images
</code></pre>
