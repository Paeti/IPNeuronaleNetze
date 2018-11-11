#Use tensorflow image as parent image
#So tensorflow, numpy, matlibplot, pip and all the other dependencies are installed
FROM tensorflow/tensorflow:latest

RUN apt-get update

#Install opencv
RUN apt-get install python-opencv -y


#Deletes the default jupiter notebooks and nn's
RUN rm *

#Install git
RUN apt-get install git -y

#Clone our repo
RUN git clone https://github.com/Paeti/IPNeuronaleNetze

RUN cd IPNeuronaleNetze
