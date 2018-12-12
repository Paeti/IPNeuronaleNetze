import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.model import Model
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    genderFilepath = "filpath to gender"
    ageFilepath = "filepath to age"
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    genderData = DataGenerator(config, genderFilepath)
    ageData = DataGenerator(config, ageFilepath)

    # create an instance of the model you want
    genderModel = Model(config, 2)
    ageModel = Model(config, 101)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    genderTrainer = Trainer(
        sess, genderModel, genderData, config, logger)
    ageTrainer = Trainer(sess, ageModel, ageData, config, logger)
    # load gender model if exists
    genderModel.load(sess)
    # here you train your gender model
    genderTrainer.train()
    # load age model if exists
    ageModel.load(sess)
    # here you train your age model
    ageTrainer.train()

    # for the saving of god
    genderModel.save('gender_model.h5')
    ageModel.save('age_model.h5')


if __name__ == '__main__':
    main()
