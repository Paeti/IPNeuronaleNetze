# Research Results

## About which neural networks fits best our needs and which tasks we'll face.

<p>
title: my research results<br>
author: Patrick Reckeweg<br>
date: 13.11.18
</p>

After doing some further research I think we focused to much on just building the network.
We even forgot to think/ talk about whether a single network architecture solves both of our
classification problems. [Here](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf) I found an interesting
paper, doing the same stuff we want to do. IN the paper is said that cnn's are best choice for face recognition.
Same is said [here](https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d).
Both the article and the link provide a mass of links to do some further research on, but I hadn't the time for doing it yet.
Reading this papers I noticed that building the network was in both cases one of the smaller project tasks.
[This](https://www.researchgate.net/publication/283356929_DEX_Deep_EXpectation_of_Apparent_Age_from_a_Single_Image) paper confirmed my adoption.
Trying to guess how long it could take to build the network I read [this](https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5)
and I would say we should be able to build the network in not more then 3 weeks(a 10 hours/person per week).
If the people are a little bit used to tensorflow and nn's it could be done in half of the time.
I think its essential, because all this papers let assume that plan the training, seperate the datasets, prepare them and train the models will take a lot more time.
We should talk with Prof. Hartung about using pretrained weights with our planned vgg-16 network.
The [team](https://www.researchgate.net/publication/283356929_DEX_Deep_EXpectation_of_Apparent_Age_from_a_Single_Image) which used vgg-16 used an model pretrained on the net datasets
on general classification. This would improve our result enormous, but would take up to 5 weeks on 4 gpu's. Because we don't have the time for that we need to use pretrained weights, choose
another nn or accept bad result.

Next steps I would choose:
- __talk to Prof Hartung about using pretrained weights(otherwise search for other fitting net)__
- research if vgg-16 is a good choice for gender estimation too
- understanding what convolutional nn's exactly do
- image group:
  - which iput vgg-16 takes
  - how to split the datasets
  - how to prepare the images(rotate or zoom them for training => better generalization)
  - get datasets
- nn group:
  - build the network
  - get in touch with tensorboard
  - play arround with smaller nn's and learn how to train(all the stuff with graphs and optimization)


<p>
title: Alex Notizen<br>
author: Alex Wiens<br>
date: 13.11.18
</p>

VGG-16 erkennt zwar gut alter und Geschlecht, für weitere Features ist es aber vielleicht sinnvoll erstmal einen Gesichts-ausschnitt zu erhalten (eventuell auch mit Augen Position um je nachdem nachher einfacher die Augenfarbe zu erhalten).
Hierfür bietet sich ein einfaches NN basierend auf "Haar-Cascades" an (also relativ einfache Kontrast Analyse zwischen verschiedenen Positionen im Bild) welches von Intel zur Verfügung gestellt wird und direkt mit OpenCV ausgeliefert wird (allerdings als separate XML Datei)

- https://github.com/shantnu/FaceDetect/blob/master/face_detect.py
- https://realpython.com/face-detection-in-python-using-a-webcam/


__english version:__
VGG-16 is good for age and gender recognition, but for further features it might be useful to get face detail first (possibly with eye position to get the eye color easier afterwards). For this a simple NN based on "hair cascades" (i.e. relatively simple contrast analysis between different positions in the image) which is provided by Intel and shipped directly with OpenCV (but as a separate XML file** is useful.

- https://github.com/shantnu/FaceDetect/blob/master/face_detect.py
- https://realpython.com/face-detection-in-python-using-a-webcam/


## __*comment on Alex research:*__
*We use the webcam via the webbrowser, I'm not shure how or if it's possible to detect faces in already saved pictures.*
*If a member of the image team needs something to research, search for this!*



<p>
title: research<br>
author: Katrin Hammacher<br>
date: 14.11.2018
</p>

[Here](https://www.analyticsvidhya.com/blog/2017/06/hands-on-with-deep-learning-solution-for-age-detection-practice-problem) you can find some instructions for age detection with deep learning, using python and keras. They also provide an introduction about implementing a [neural network with tensorflow](https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow) and some further informations about age detection and the applications we want to work with.

On most of the websites I went through (like [this](https://blog.statsbot.co/neural-networks-for-beginners-d99f2235efca) one) they tell you to use a convolutional neural network, for face detection. [Here](https://github.com/mks0601/A-Convolutional-Neural-Network-Cascade-for-Face-Detection) is an example. 

VGG-16 is a suitable [architecture for Large-scale image processing](https://www.jeremyjordan.me/convnet-architectures) and is named as a very deep convolutional network. You can find an example for a VGG16 model for Keras [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).

<p>
title: Ronny Notizen<br>
author: Ronny Aretz<br>
date: 15.11.18
</p>

In my opinion we should use vgg 16 to reduce the development time.

After reading a few search results, i have noticed that other projekts used keras instead of tensorflow. I think vgg 16 is easy portable to tensorflow. But we should talk about that.

But I think its important to know that vgg 16 don't guess the age. So wie must input some layer and train this model for that i think. But its possible that i am wrong.

sorry for my bad english...