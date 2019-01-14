# encoding=utf8
import time
import pickle

import tensorflow as tf
from model import Model
from utils import get_logger, create_model
from utils import load_config
from data_utils import load_word2vec, input_from_line

flags = tf.app.flags


flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("config_file",  "config_file",  "File for config")


FLAGS = tf.app.flags.FLAGS


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                start = time.time()
                print(input_from_line(line, char_to_id))
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                end = time.time()
                print(result, (end - start)*1000)


if __name__ == "__main__":
    evaluate_line()



