#The "*fg-script.py*" file will -

<pre><code>
1. look if a file exists at *"../data/FGNET.zip"* exists - do nothing; else - download dataset from
"*http://yanweifu.github.io/FG_NET_data/FGNET.zip"*

2. look if an extracted file at *"../data/data/FGNET"* exists - do nothing; else - extract downloaded dataset

3. have a function to create Classification folders in *"../data/LAP"* ranging from 0 to 100 (not used)

4. open images and creates an array of tupels with the corresponding age value which is extracted from
the image name

5. write the images into the fitting Classification folder

6. create a *"img_counts_fg.txt"* file with the amount of Training, Validiation and Test images
</code></pre>