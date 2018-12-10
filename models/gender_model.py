from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers


class Model:
    def __init__(self, filepath, classes):
        self.main_model(filepath, classes)

    def main_model(filepath, classes):
        #STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCHSIZE
        image, label = create_dataset(filepath)

        train_model = build_model()

        train_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                            loss='mean_squared_error',
                            metrics=[soft_acc],
                            target_tensors=[label])

        train_model.fit(epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOC)
        init_saver(train_model)

    def build_model(input_shape=(3, 224, 224), classes=2):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape))
        model.add(Convolution2D(64, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, LeakyReLU(alpha=0.3)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(4096, LeakyReLU(alpha=0.3)))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model

    def create_dataset(filepath):
        dataset = tf.data.TFRecordDataset(filepath)
        dataset = dataset.map(_parse_function, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()
        image, label = iterator.get_next()
        image = tf.reshape(image, [-1, 224, 224, 1])
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label

    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                            "label": tf.FixedLenFeature([], tf.int64)}
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)
        # Turn your saved image string into an array
        parsed_features['image'] = tf.decode_raw(
            parsed_features['image'], tf.uint8)
        return parsed_features['image'], parsed_features["label"]

    def init_saver(model):
        model.save('gender_model.h5')
