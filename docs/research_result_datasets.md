<p>
title: research<br>
author: Katrin Hammacher<br>
date: 13.11.18
</p>


For a better comparability we should split our dataset in a test and a training dataset. The
classificator is getting trained on the training set. The results can be tested
on the test set. You can find further information in
[english](https://machinelearningmastery.com/difference-test-validation-datasets) and
[german](http://www.is.informatik.uni-wuerzburg.de/fileadmin/10030600/Mitarbeiter/Reul_Christian/Baumklassifikation_Reul_Christian_MA.pdf)
(especially 2.2.3.1).
Important is that the test and training set are from the same distribution and are taken
randomly.
It has to be big enough to be representable (for >1M examples --> 10.000 examples each).

There is also the possibility to work with [three different
groups](https://cs230-stanford.github.io/train-dev-test-split.html#theory-how-to-choose-the-train-train-dev-dev-and-test-sets)
(On that website is also a short example how to implement a splitting of the data):
test, development/validation and training set. The reason having one more set is to get a high
performance neuronal network. The training algorithms are getting trained training
set. To find out with algorithm performs best we let them run over the dev/val set. The final
algorithm can be tested on the test set.

More Links:
https://www.coursera.org/lecture/deep-neural-network/train-dev-test-sets-cxG1s
https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data
