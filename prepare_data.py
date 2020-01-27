import numpy as np
import cv2
from xml.dom.minidom import parse

def _to_float(strs):
    return [float(s) for s in strs]

def _read_points(trace):
    points = [_to_float(point.strip().split()[:2])
              for point in trace.split(',')]
    return points

def read_strokes_inkml(inkmlfile):
    try:
        rootInk = parse(inkmlfile)
        traces = [_read_points(trace.firstChild.nodeValue.strip())
                  for trace in rootInk.getElementsByTagName('trace')]
        return traces
    except:
        print('Exception:{}'.format(inkmlfile))

def create_vocab():
    sup_data = read_supervised_data()
    latex = list(zip(*sup_data))[1]
    latex = [latex_str.strip().split() for latex_str in latex]

    from itertools import chain

    latex_syms = list(set(chain.from_iterable(latex)))
    latex_syms.sort()

    with open('vocab_syms.txt', 'w') as f:
        f.writelines([latex_sym + '\n' for latex_sym in latex_syms])

    print (len(latex_syms))

def check_vocab():
    sup_data = read_supervised_data()
    latex = list(zip(*sup_data))[1]
    latex = [latex_str.strip().split() for latex_str in latex]

    from itertools import chain

    latex_syms = list(set(chain.from_iterable(latex)))

    vocab = [line.strip() for line in open('vocab_syms.txt').readlines()]

    [print(sym) for sym in latex_syms if sym not in vocab ]

def read_supervised_data(fileName):
    with open(fileName, 'r') as f:
        sup_data =[line.strip().split('\t') for line in f.readlines()]
    return sup_data

def create_supervised_label(batch_latex, vocab):
    # print (batch_latex)
    idx_to_vocab = dict([(i, vocab[i]) for i in range(len(vocab))])
    vocab_to_idx = dict([(vocab[i], i) for i in range(len(vocab))])

    labels = [[vocab_to_idx[sym] for sym in latex] for latex in batch_latex]

    batch_target = np.zeros((len(labels), len(vocab)))

    for i, label in enumerate(labels):
        batch_target[i, label] = 1

    return batch_target

def read_image(file_name, image_shape):
    image = cv2.imread(file_name, 0) # gray scale
    img = cv2.resize(image, image_shape)
    img = np.array(img).astype(float)
    return img

def create_input_features(input_file):
    data_path = 'D:\\database\\CROHME\\CROHME_2016_for_clustering\\png\\train'

def create_supervised_data(num_samples = 100):

    annotation = 'crohme2016_train_annotation_sym_seq.txt'

    vocab = [line.strip() for line in open('vocab_syms.txt').readlines()]

    sup_data = read_supervised_data(annotation)

    # import random
    #
    # random.shuffle(sup_data)

    sup_data = sup_data[:]

    input_files, latex = list(zip(*sup_data))
    latex = [latex_str.strip().split() for latex_str in latex]

    label_seqs = create_supervised_label(latex[:100], vocab)

    # print(input_files)

    print('label seqs: ', len(label_seqs))

    input_seqs = list(map(create_input_features, input_files))

    print('input seqs: ', len(input_seqs))
    return input_seqs, label_seqs

def create_annotation_sym_set():
    annotation = 'crohme2016_test_annotation_sym_seq.txt'
    # vocab = [line.strip() for line in open('vocab_syms.txt').readlines()]
    sup_data = read_supervised_data(annotation)

    input_files, latex = list(zip(*sup_data))
    latex = [sorted(list(set(latex_str.strip().split()))) for latex_str in latex]

    with open('crohme2016_test_annotation_sym_set.txt', 'w') as f:
        f.writelines([file + '\t' + ' '.join(sym_set) + '\n' for file, sym_set in zip(input_files, latex)])

def count_formula():
    annotation = 'crohme2016_train_annotation_sym_set.txt'
    sup_data = read_supervised_data(annotation)
    annotation = 'crohme2016_test_annotation_sym_set.txt'
    sup_data += read_supervised_data(annotation)

    input_files, sym_sets = list(zip(*sup_data))

    from collections import Counter

    c = Counter(sym_sets)
    print(c)

def main():
    count_formula()
    return
main()