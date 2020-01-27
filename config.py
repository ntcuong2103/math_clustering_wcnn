
DATASET = "2016"
# DATASET = "2019"

CONFIGURATION =  {
    "VOCAB_PATH": 'vocab_syms.txt',
    "MODEL_PATH": './models/',
    "LOG_PATH":  './logs/',
    "CUDA_DEVICE": "1",
    "MODEL_NAME": "cnn_multiscale_3lvl_ft2_gattp_gmp",
    "INITIAL_EPOCH": 201,
    "BATCH_SIZE": 1,
    "LRATE": 1e-3,

    "NUM_EPOCH": 400,
    "NUM_OUT": 101,
    "NUM_CHANNEL": 1,

    # bool value
    "IS_MULTI_GPU": False,
    "IS_ATTENTIVE": False,
    "IS_TRAINED": False,

    # no need to change here
    "TRAIN_DATA_PATH": './CROHME' + DATASET + '/png/train',
    "TRAIN_ANNOTATION": 'crohme' + DATASET + '_train_annotation_sym_seq.txt',

    "VAL_DATA_PATH": './CROHME' + DATASET + '/png/valid',
    "VAL_ANNOTATION": 'crohme' + DATASET + '_valid_annotation_sym_seq.txt',

    "TEST_DATA_PATH": 'D:/Workspace/math_clustering_wcnn/data/Khuong/Dset_Mix',
    "TEST_ANNOTATION": 'annotations/dset_mix_filelist.txt',

    "TRAIN_CROSS_DATA_PATH": './math_collect/sasaki/imgs',
    "TRAIN_CROSS_ANNOTATION": 'sasaki_annotation_sym_seq_cross.txt',
    # "TRAIN_CROSS_DATA_PATH": '/home/tuancuong/math_collect/khuong/imgs_h128',
    # "TRAIN_CROSS_ANNOTATION": 'khuong_annotation_sym_seq_cross.txt',

    "VAL_CROSS_DATA_PATH": '/home/tuancuong/math_collect/sasaki/imgs',
    "VAl_CROSS_ANNOTATION": 'sasaki_annotation_sym_seq_cross.txt',
}