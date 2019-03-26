## IPNeuronaleNetze

We are currently building a VGG-16 convolutional neural network. It'll be trained for age and gender estimation.
For better usage of the trained model a webserver and REST API will be built.

## Installation
Just run the following in a shell
>  curl -fsSL https://gist.github.com/DZvO/00fd8a496050547c2f89cd634f47283e/raw/9d61d503cc5aaf995590dc83bb9afa7abe6bbdc6/install.sh | sudo bash -s  

Then open [https://localhost:5000/#](https://localhost:5000/#)

## Motivation

We're all interested in ai and neural networks and want to learn something about how to implement one.
So we're happy that we could choose this as a project at our university, the FH Aachen.


## Code style

[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)


## Tech/framework used

<b>Built with</b>
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Opencv](https://opencv.org/)
- [Flask](http://flask.pocoo.org/)
- [Docker](https://www.docker.com/)
- [Drone CI](https://drone.io/)

Tensorflow and keras are used to build the neural network and to train it.
Opencv is usesd to prepare the images.
We are using Flask for setting up a simple REST API. Which handles the interaction with our frontend.
Docker is used to simplify the delivery.
Continous integration tool of our choice is drone ci.


## Contribute

We are happy when you're interested in our project. If you want to contribute read our [contributing guideline](CONTRIBUTING.md).


## Credits

Project structure inspired by [this repo](https://github.com/MrGemy95/Tensorflow-Project-Template).
Contribution guideline inspired by [this one](https://github.com/angular/angular.js/blob/master/CONTRIBUTING.md#commit).


## License

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

MIT © [Patrick Reckeweg](https://github.com/Paeti) [Alexander Wiens](https://github.com/DZvO) [Bram Wigger]() [Christopher Kremkow]() [Jonas Kau]() [Katrin Hammacher]() [Max Schmidt]() [Ronny Aretz]()
