"""
deprecated ?

"""
from config import CONFIGURATION
import numpy as np
from PIL import Image
import random
import cv2
from keras.applications import imagenet_utils
from data_utils import read_supervised_data, create_supervised_label

def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='tf')

def load_imgsize(file_list):
    def get_size(data):
        img = Image.open(data)
        return img.size[0]
    train = list(map(get_size, file_list))
    return train

def sort_by_images(images):
    idx_imgsize = [(idx, image[0]) for idx, image in enumerate(images)]
    sorted_imgsize = sorted(idx_imgsize, key=lambda x: x[1], reverse=True)
    sorted_idx, _  = list(zip(*sorted_imgsize))
    images = [images[idx] for idx in sorted_idx]
    return images, np.array(sorted_idx)

def read_image(file_name, image_shape, dilation=2):
    image = cv2.imread(file_name, 0) # gray scale

    # processing for Khuong data
    image = 255 - image # inverse
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT) # padding
    image = cv2.dilate(image, kernel = np.ones((dilation,dilation),np.uint8), iterations=3)

    img = cv2.resize(image, image_shape)
    img = np.array(img).astype(float)
    return img

def Batch(data, out_height=128, max_width=800, shuffle = False, is_train=False, batch_size=1, preprocess_input=preprocess_input):
    L = len(data)
    min_width = 32
    while True:
        batch_starts = np.arange(0, L - batch_size + 1, batch_size)
        if shuffle:
            random.shuffle(data)

        for batch_start in batch_starts:
            batch_input_images = list()
            labels = list()

            batch_img_width = int (data[batch_start][0])
            if batch_img_width > max_width:
                batch_img_width = max_width

            if batch_img_width < min_width:
                batch_img_width = min_width

            for id in range (batch_size):
                file_name = data[batch_start+id][1]
                # print (batch_start, file_name)
                # read image
                image = read_image(file_name, (batch_img_width, out_height))
                file_class = data[batch_start+id][2]

                batch_input_images.append(image)
                labels.append(file_class)

            batch_input_images = np.array(batch_input_images)
            batch_input_images = np.expand_dims(batch_input_images, -1)

            yield (preprocess_input(batch_input_images), np.array(labels))
        if not is_train:
            break

def create_supervised_data(data_path, annotation, vocab_file):

    vocab = [line.strip() for line in open(vocab_file).readlines()]

    sup_data = read_supervised_data(annotation)

    sup_data = sup_data[:]

    filepath_latex = list(zip(*sup_data))

    filepath = filepath_latex[0]

    if len(filepath_latex) > 1:
        latex = filepath_latex[1]
    else:
        latex = [''] * len(filepath)


    # weakly label
    latex = [latex_str.strip().split() for latex_str in latex]
    weakly_labels = create_supervised_label(latex, vocab)

    # file
    filepath = [data_path + '/' + filename.replace('\\', '/') + '.png' for filename in filepath]
    img_sizes = load_imgsize(filepath)

    filepath_labels = list(zip(img_sizes, filepath, weakly_labels))

    data, _ = sort_by_images(filepath_labels)

    return data

def prepare_data_train(batch_size):
    data_path = CONFIGURATION["TRAIN_DATA_PATH"]
    annotation = CONFIGURATION["TRAIN_ANNOTATION"]
    vocab = CONFIGURATION["VOCAB_PATH"]

    train = create_supervised_data(data_path, annotation, vocab)[:]
    train_gen = Batch(train, shuffle=True, is_train=True, batch_size=batch_size)

    return train_gen, len(train)

def prepare_data_valid(batch_size):
    data_path = CONFIGURATION["VAL_DATA_PATH"]
    annotation = CONFIGURATION["VAL_ANNOTATION"]
    vocab = CONFIGURATION["VOCAB_PATH"]

    val = create_supervised_data(data_path, annotation, vocab)
    val_gen = Batch(val, shuffle=False, is_train=True, batch_size=batch_size)

    return val_gen, len(val)

def prepare_data_test(batch_size):
    data_path = CONFIGURATION["TEST_DATA_PATH"]
    annotation = CONFIGURATION["TEST_ANNOTATION"]
    vocab = CONFIGURATION["VOCAB_PATH"]

    test = create_supervised_data(data_path, annotation, vocab)
    test_gen = Batch(test, shuffle=False, is_train=False, batch_size=batch_size)

    return test_gen, len(test), test


def create_supervised_data_cross(data_path, annotation, vocab_file, filter_ids):

    vocab = [line.strip() for line in open(vocab_file).readlines()]

    sup_data = read_supervised_data(annotation)

    sup_data = sup_data[:]

    sup_data = [data for data in sup_data if int(data[2]) in filter_ids]

    filepath, latex, _ = list(zip(*sup_data))

    # weakly label
    latex = [[sym for sym in latex_str.strip().split() if sym in vocab] for latex_str in latex]

    weakly_labels = create_supervised_label(latex, vocab)

    # file
    filepath = [data_path + '/' + filename.replace('\\', '/') + '.png' for filename in filepath]
    img_sizes = load_imgsize(filepath)

    filepath_labels = list(zip(img_sizes, filepath, weakly_labels))

    filepath_labels, _ = sort_by_images(filepath_labels)

    return filepath_labels

def prepare_data_train_cross(batch_size, cross_id):
    train_cross_idx = [i for i in range(5) if i != cross_id]
    val_cross_idx = train_cross_idx[0]
    train_cross_idx = train_cross_idx[1:]
    print (train_cross_idx, val_cross_idx)

    data_path = CONFIGURATION["TRAIN_CROSS_DATA_PATH"]
    annotation = CONFIGURATION["TRAIN_CROSS_DATA_ANNOTATION"]
    vocab = CONFIGURATION["VOCAB"]

    train = create_supervised_data_cross(data_path, annotation, vocab, train_cross_idx)
    train_gen = Batch(train, shuffle=True, is_train=True, batch_size=batch_size)

    return train_gen, len(train)

def prepare_data_valid_cross(batch_size, cross_id):
    train_cross_idx = [i for i in range(5) if i != cross_id]
    val_cross_idx = [train_cross_idx[0]]
    train_cross_idx = train_cross_idx[1:]

    data_path = CONFIGURATION["VAL_CROSS_DATA_PATH"]
    annotation = CONFIGURATION["VAL_CROSS_DATA_ANNOTATION"]
    vocab = CONFIGURATION["VOCAB"]

    val = create_supervised_data_cross(data_path, annotation, vocab, val_cross_idx)
    val_gen = Batch(val, shuffle=False, is_train=True, batch_size=batch_size)

    return val_gen, len(val)



# if __name__ == '__main__':
#
#     train_gen, len_train = prepare_data_train(10)
#     for input, label in train_gen:
#         print(input.shape, label.shape)
#     print (len_train)

