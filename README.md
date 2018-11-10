## IPNeuronaleNetze

We are currently building a VGG-16 convolutional neural network. It'll be trained for age and gender estimitation.
For better usage of the trained model a webserver and REST API will be built.


## Motivation

We're all interested in ai and neural networks and want to learn something about how to implement one.
So we're happy that we could choose this as a project at our university, the FH Aachen.


## Build status

[![Build Status](http://136.243.36.114/api/badges/paeti/IPNeuronaleNetze/status.svg)](https://drone.io)


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

<!-- Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus. -->


## License

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

MIT © [Patrick Reckeweg]()
