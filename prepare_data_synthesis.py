"""
deprecated ?

"""

import numpy as np
from PIL import Image
import random
import cv2
import glob
from keras.applications import imagenet_utils
from prepare_data import read_supervised_data, create_supervised_label

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
    # image = 255 - image # inverse
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT) # padding
    image = cv2.erode(image, kernel = np.ones((dilation,dilation),np.uint8), iterations=1)


    img = cv2.resize(image, image_shape)
    img = np.array(img).astype(float)
    return img

def Batch(data, out_height=128, max_width=700, shuffle = False, is_train=False, batch_size=1, preprocess_input=preprocess_input):
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

def create_supervised_data(data_path, annotation, vocab_file, sort=True):

    vocab = [line.strip() for line in open(vocab_file).readlines()]

    sup_data = read_supervised_data(annotation)

    sup_data = sup_data[:]

    filepath, latex = list(zip(*sup_data))

    # weakly label
    latex = [[sym for sym in latex_str.strip().split() if sym in vocab] for latex_str in latex]

    weakly_labels = create_supervised_label(latex, vocab)

    # file
    filepath = [data_path + '\\' + filename + '.png' for filename in filepath]
    img_sizes = load_imgsize(filepath)

    filepath_labels = list(zip(img_sizes, filepath, weakly_labels))

    if sort:
        filepath_labels, _ = sort_by_images(filepath_labels)

    return filepath_labels

def prepare_data_train(batch_size):
    data_path = 'D:\\database\\CROHME\\CROHME_2016_for_clustering\\png\\train'
    annotation = 'crohme2016_train_annotation_sym_seq.txt'

    vocab = 'vocab_syms.txt'

    train = create_supervised_data(data_path, annotation, vocab)
    train_gen = Batch(train, shuffle=True, is_train=True, batch_size=batch_size)

    return train_gen, len(train)

def prepare_data_valid(batch_size):
    data_path = 'D:\\database\\CROHME\\CROHME_2016_for_clustering\\png\\valid'
    annotation = 'crohme2016_valid_annotation_sym_seq.txt'
    vocab = 'vocab_syms.txt'

    val = create_supervised_data(data_path, annotation, vocab)
    val_gen = Batch(val, shuffle=False, is_train=True, batch_size=batch_size)

    return val_gen, len(val)

def prepare_data_test(batch_size):
    # CROHME
    data_path = 'D:\\database\\CROHME\\CROHME_2016_for_clustering\\png\\test'
    annotation = 'crohme2016_test_annotation_sym_seq.txt'

    # CROHME 2019
    data_path = 'D:\\database\\CROHME\\CROHME_2019\\png\\train'
    annotation = 'crohme2019_test_annotation_sym_seq.txt'

    # sasaki
    # data_path = 'D:\\Workspace\\math_clustering_wcnn\\data\\Sasaki\\imgs'
    # annotation = 'sasaki_test_annotation_group.txt'

    # khuong
    # data_path = 'D:\\Workspace\\math_clustering_wcnn\\data\\Khuong\\imgs_h128'
    # annotation = 'khuong_test_annotation_group.txt'

    # custom test
    # data_path = 'D:\\database\\CROHME\\CROHME_2016_for_clustering\\png\\viz'
    # annotation = 'viz_annotation_sym_seq.txt'

    vocab = 'vocab_syms.txt'

    test = create_supervised_data(data_path, annotation, vocab)[:]
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
    filepath = [data_path + '\\' + filename + '.png' for filename in filepath]
    img_sizes = load_imgsize(filepath)

    filepath_labels = list(zip(img_sizes, filepath, weakly_labels))

    filepath_labels, _ = sort_by_images(filepath_labels)

    return filepath_labels

def prepare_data_train_cross(batch_size, cross_id):
    train_cross_idx = [i for i in range(5) if i != cross_id]
    val_cross_idx = train_cross_idx[0]
    train_cross_idx = train_cross_idx[1:]

    data_path = 'data\\Khuong\\imgs_h128'
    annotation = 'khuong_annotation_sym_seq_cross.txt'
    vocab = 'vocab_syms.txt'

    train = create_supervised_data_cross(data_path, annotation, vocab, train_cross_idx)
    train_gen = Batch(train, shuffle=True, is_train=True, batch_size=batch_size)

    return train_gen, len(train)

def prepare_data_valid_cross(batch_size, cross_id):
    train_cross_idx = [i for i in range(5) if i != cross_id]
    val_cross_idx = [train_cross_idx[0]]
    train_cross_idx = train_cross_idx[1:]

    data_path = 'data\\Khuong\\imgs_h128'
    annotation = 'khuong_annotation_sym_seq_cross.txt'
    vocab = 'vocab_syms.txt'

    val = create_supervised_data_cross(data_path, annotation, vocab, val_cross_idx)
    val_gen = Batch(val, shuffle=False, is_train=True, batch_size=batch_size)

    return val_gen, len(val)

def prepare_data_test_cross(batch_size, cross_id):
    test_cross_idx = [cross_id]

    # data_path = 'data\\Khuong\\imgs_h128'
    # annotation = 'khuong_annotation_sym_seq_cross.txt'

    data_path = 'data\\Sasaki\\imgs'
    annotation = 'sasaki_annotation_sym_seq_cross.txt'
    vocab = 'vocab_syms.txt'

    test = create_supervised_data_cross(data_path, annotation, vocab, test_cross_idx)[:]
    test_gen = Batch(test, shuffle=False, is_train=False, batch_size=batch_size)

    return test_gen, len(test), test



def check_latex():
    data_list = 'FileName_GroundTruth_List.txt'
    lines = open(data_list).readlines()


    latex = [map(lambda  s: s.replace('\\\\', '\\').replace('<', '\\lt').replace('>', '\\gt'), line.strip().split('\t')[1].split()) for line in lines]

    from itertools import chain


    latex_syms = list(set(chain.from_iterable(latex)))
    print (latex_syms)

    vocab = [line.strip() for line in open('vocab_all_fixed.txt').readlines()]

    print ([i for i in latex_syms if i not in vocab])

if __name__ == '__main__':
    # data_list = './data_v/Khuong/khuong-offMEclus-data-1line/file_list.txt'
    # test_gen, len_test, test = prepare_data_test_Khuong_Sasaki(data_list)

    # prepare_data_train_Sasaki()
    # divide_train_val()

    train_gen, len_train, _ = prepare_data_test(10)
    for input, label in train_gen:
        print(input.shape, label.shape)
    #
    print (len_train)
    # check_latex()

    # # data_list = './data_v/Sasaki/file_list.txt'
    # data_size = load_imgsize_Khuong_Sasaki(data_list)
    # print (len(data_size))
    # # print (data_size)
    # cluster_list = list(set([get_cluster_label_Khuong(cluster_label[1]) for cluster_label in data_size]))
    #
    # print (len(cluster_list))


    # # prepare and save data
    # # prepare_data_train()
    # # prepare_data_test(10)
    #
    # # train_gen, val_gen, train_len, val_len = prepare_data_pretrain()
    # # print (train_len, val_len)
    # #
    # # for imgs, lbls in train_gen:
    # #     print (imgs.shape, lbls.shape)
