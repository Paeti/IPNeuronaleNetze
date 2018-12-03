import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso



class Model:
    def __init__(self, config):
        self.fit_model()
        self.build_model()
    #**
    # Hier könnte man die input/Image größe oder die reihenfolge der Features
    # selbst noch ändern aber default ist die von VGG 16
    #**
    def build_model(self,input_shape=(3,224,224)):
        #**
        # Ich habe die python implementation von vgg 16 genommen und
        # bis auf die Aktivierungsfunktinen nichts geändert
        #**
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape))
        #**
        # Ich habe mich für die Aktivierungsfunktion LeakyRelu (vorher einfache Relu)
        # entschieden um dem Problem des "dying relu" zu umgehen.
        # Im grunde optimiert die LeakyRelu nur das gradienten verfahren welches
        # zur Fehlerminimierung genutzt wird.
        # Genaueres siehe: https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning
        #**
        model.add(Convolution2D(64, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        #**
        # Hier ist der hintergedanke das wir zwei werte zwischen 0 und 99 bekommen
        # ein Wert für das alter von 0 bis 99 Jahre
        # und ein Wert für das Geschlecht z.B. zwischen 0 - 49 für eine Frau
        # und von 50 - 99 für einen Mann
        #**
        model.add(Dense(2, relu(99, alpha=0.0, max_value=None, threshold=0.0)))

        return model

    #**
    # https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    #
    # Parameter cv ist die anzahl der einteilung des datasets
    # Parameter dataset ist aufgeteilt in einer zeilenvector matrix "data" die die featurespixel enthalten
    # und eine spaltenvector matrix "target" mit den tags
    # hier müsste angepasst werden wenn das dataset anders aufgebaut ist
    # 
    # I mache hier eine Random search mit verschiedenen parametern und gleichzeitig eine crossvalidation
    # am ende gebe ich die ergebnisse aus
    #**
    def fit_model(self, dataset, cv = 10):
        param_dist = {'weights': ['uniform', 'distance'], 'p': sp_randint(1, 2)}
        n_iter_search = 100
        param_grid = {'alpha': sp_rand()}
        model = self.build_model()
        rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100,cv=cv)
        rsearch.fit(dataset.data, dataset.target)
        print(rsearch)
        # summarize the results of the random parameter search
        print(rsearch.best_score_)
        print(rsearch.best_estimator_.alpha)

        pass