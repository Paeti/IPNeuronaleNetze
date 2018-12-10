from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
 class Model:
    def __init__(self, config):
        self.fit_model()
        self.build_model()
     # Load the VGG model
    #**
    # Ich habe die python implementation von vgg 16 genommen und
    # bis auf die Aktivierungsfunktinen nichts geändert
    #**
    def build_model(input_shape=(3,224,224),klassen=100):
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
        model.add(Dense(klassen, activation='softmax'))
        return model
     def fit_model(self, dataset, cv = 10):
        image_size = 224
         # Training and Validation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
         validation_datagen = ImageDataGenerator(rescale=1./255)
             # Change the batchsize according to your system RAM
        train_batchsize = 100
        val_batchsize = 10
            
        train_generator = train_datagen.flow_from_directory(
            directory= r"C:\Users\ckrem\Desktop\IP\Data\Train" ,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical')
        
        validation_generator = validation_datagen.flow_from_directory(
            directory= r"C:\Users\ckrem\Desktop\IP\Data\Valid",
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)
         # Compile the model
        model = self.build_model()
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-4),
                    metrics=['acc'])
        # Train the model
        # generator is used when you want to avoid duplicate data when using multiprocessing. This is for practical purpose, when you have large dataset.
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples/train_generator.batch_size ,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples/validation_generator.batch_size,
            verbose=1)
        
        # Save the model
        model.save('age_model.h5')
        # Plot training and validation data
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(len(acc))
        
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()