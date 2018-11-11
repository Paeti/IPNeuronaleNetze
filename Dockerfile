#Use tensorflow image as parent image
#So tensorflow, numpy, matlibplot, pip and all the other dependencies are installed
FROM tensorflow/tensorflow:latest

RUN apt-get update \
  && apt-get install \
     python-opencv \
     git -y

WORKDIR /

#Deletes the default jupiter notebooks and nn's
RUN rm -rf /notebooks/* \
  && git clone https://github.com/Paeti/IPNeuronaleNetze /notebooks/IPNeuronaleNetze

WORKDIR /notebooks/IPNeuronaleNetze
