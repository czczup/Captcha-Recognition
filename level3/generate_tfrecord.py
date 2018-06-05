import tensorflow as tf
import os
import random
import math
import sys
from PIL import Image
import numpy as np
import conf

# The number of images in the validation set.
_NUM_VALID = 10000

def _get_filenames_and_classes(dataset_dir):
    """ Get all classes and filenames. """
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir,filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    image_filenames = []

    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory,filename)
            image_filenames.append(path)

    return image_filenames,class_names

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image':bytes_feature(image_data),
        'label':int64_feature(class_id),
    }))

def write_label_file(labels_to_class_names,labels_filename):
    """ Write value-label into labels.txt """
    with tf.gfile.Open(labels_filename,"w") as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label,class_name))

def _convert_dataset(split_name,filenames,class_names_to_ids,dataset_dir):
    """ Convert data to TFRecord format. """
    assert split_name in ['train','valid']
    with tf.Graph().as_default():
        with tf.Session() as sess:
            output_filename = os.path.join(dataset_dir,"image_%s.tfrecord" % split_name)
            tfrecord_writer = tf.python_io.TFRecordWriter(output_filename)
            for i in range(len(filenames)):
                image_data = Image.open(filenames[i])
                image_data = image_data.resize((28, 40))
                image_data = np.array(image_data.convert('L'))
                image_data = image_data.tobytes()
                class_name = os.path.basename(os.path.dirname(filenames[i]))
                class_id = class_names_to_ids[class_name]
                example = image_to_tfexample(image_data,class_id)
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1,len(filenames)))
                sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    # Get all images and classes.
    image_filenames, class_names = _get_filenames_and_classes(conf.CUT_PATH)
    # Change classes into Python dict
    class_names_to_ids = dict(zip(class_names,range(len(class_names))))
    # Shuffle the image set.
    random.seed(0)
    random.shuffle(image_filenames)
    # Divide the data into training sets and validation sets.
    training_filenames = image_filenames[_NUM_VALID:]
    validation_filenames = image_filenames[:_NUM_VALID]
    # Convert data to TFRecord format.
    _convert_dataset('train', training_filenames, class_names_to_ids, conf.TFRECORD_PATH)
    _convert_dataset('valid', validation_filenames, class_names_to_ids, conf.TFRECORD_PATH)
    # Generate labels.txt.
    labels_to_class_names = dict(zip(range(len(class_names)),class_names))
    write_label_file(labels_to_class_names, conf.LABEL_PATH)

