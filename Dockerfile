#Use tensorflow image as parent image
#So tensorflow, numpy, matlibplot, pip and all the other dependencies are installed
FROM tensorflow/tensorflow:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install \
     python-opencv \
     python3-opencv \
     git -y

WORKDIR /

#Deletes the default jupiter notebooks and nn's
RUN rm -rf /notebooks/*

WORKDIR /notebooks
