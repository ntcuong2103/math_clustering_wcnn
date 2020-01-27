"""
2019/01
@author　Ushizawa
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import multi_gpu_model

from datetime import datetime
from CNN_multiscale_3lvl_2ft_pretrain_diag_viz_multiloss import buildVisualizeModel2 as buildModel
from CNN_multiscale_3lvl_2ft_pretrain_diag_viz_multiloss import buildLocalAttentiveModel as buildModel

from prepare_data_synthesis import prepare_data_train, prepare_data_valid, prepare_data_test
from prepare_data_synthesis import prepare_data_train_cross, prepare_data_valid_cross, prepare_data_test_cross

MODEL_PATH = './models/'
LOG_PATH = './logs/'


# train_data_file = './data_v/npz/img-PHOC_CROHME_train_lv1_nopad.npz'
# test_data_file = './data_v/npz/img-PHOC_CROHME_test_lv1_nopad.npz'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


def recall(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    c = a * b
    return K.sum(c) / K.sum(a)

# def recall_diag(y_true, y_pred):
#     a = y_true >= 0.5
#     b = y_pred >= 0.5
#     a = K.cast(a, K.floatx())
#     b = K.cast(b, K.floatx())
#     c = a * b
#     return K.sum(c) / K.sum(a)

def precision(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    c = a * b
    return K.sum(c) / (K.sum(b) + 0.0001)

def fmeasure(y_true, y_pred):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)

    f = 2.0 * r * p / (r + p + 0.0001)
    return f

def precision_diag(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    mask = tf.expand_dims(tf.reduce_max(y_true, axis=-1), -1)

    c = a * b
    return K.sum(c) / (K.sum(b * mask) + 0.01)

def recall_np(y_true, y_pred):
    a = np.array(y_true >= 0.5).astype(float)
    b = np.array(y_pred >= 0.5).astype(float)

    c = a * b
    return np.sum(c) / np.sum(a)

def precision_np(y_true, y_pred):
    a = np.array(y_true >= 0.5).astype(float)
    b = np.array(y_pred >= 0.5).astype(float)

    c = a * b
    return np.sum(c) / (np.sum(b) + 1.0)

def binary_entropy_drop(y_true, y_pred):
    drop_rate = 0.0

    return K.sum(K.dropout(K.binary_crossentropy(y_true, y_pred), drop_rate), axis=-1) #/ (1 - drop_rate + K.epsilon())

def binary_entropy_class_drop(y_true, y_pred):
    drop_rate = 0.0
    # mask_loss = tf.stack(tf.reduce_max(y_true, axis=-1) * y_true.shape[-1], axis=-1)
    mask_loss = tf.expand_dims(tf.reduce_max(y_true, axis=-1), -1)
    return K.sum(K.mean(mask_loss * K.binary_crossentropy(y_true, y_pred), axis=-1), axis=-1) #/ (1 - drop_rate + K.epsilon())

class CNN_build_model():

    def __init__(self):
        # パラメータの設定
        self.num_epoch = 400
        channels = 1
        # n_out = 97
        n_out = 101

        self.model = buildModel(input_shape=[None, None, channels], n_out=n_out)
        self.model.summary()

        # self.model = multi_gpu_model(model=self.model)

        self.model.compile(loss=binary_entropy_drop, optimizer=Adam(lr=0.001),
                           metrics=[recall, precision, fmeasure])

    # 形式に合わせて学習する
    def fit(self, features, likelihoods, batch_size = 1, initial_epoch=0):
        save_prefix = self.model.name + '_CROHME_small_phoclvl1'

        callbacks = [EarlyStopping('val_recall', patience=50, mode='max'),
                     ModelCheckpoint(filepath=MODEL_PATH + save_prefix + '.ep{epoch:03d}.h5',
                                     monitor='val_recall',
                                     save_best_only=True, mode='max'),
                     CSVLogger(LOG_PATH + save_prefix + "@" + datetime.now().strftime('%Y.%m.%d-%H.%M.%S') + '.csv')]
        return self.model.fit(features, likelihoods, validation_split=0.1, epochs=self.num_epoch,
                              batch_size=batch_size, callbacks=callbacks,
                              verbose=2, initial_epoch=initial_epoch)

    def fit_generator(self, train_generator, steps_per_epoch, validation_generator,
                      validation_steps, initial_epoch=0):
        save_prefix = self.model.name + '_CROHME_2016'

        callbacks = [EarlyStopping('val_fmeasure', patience=200, mode='max'),
                     ModelCheckpoint(filepath=MODEL_PATH + save_prefix + '.ep{epoch:03d}.h5',
                                     monitor='val_fmeasure',
                                     save_weights_only=True,
                                     save_best_only=True, mode='max'),
                     CSVLogger(LOG_PATH + save_prefix + "@" + datetime.now().strftime('%Y.%m.%d-%H.%M.%S') + '.csv')]

        return self.model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=self.num_epoch,
                                        verbose=1, callbacks=callbacks, validation_data=validation_generator,
                                        validation_steps=validation_steps, max_queue_size=1, shuffle=False, initial_epoch=initial_epoch)


    def save_weights(self):
        self.model.save_weights('PHOC_train.h5')

    # 重さを初期化する関数
    def weight_variable(self, shape):
        return K.truncated_normal(shape, stddev=0.01)

    # 検証用
    def predict_proba(self, samples):
        return self.model.predict(samples, batch_size=self.mini_batch_size)

    def evaluate(self, features, likelihoods):
        return self.model.evaluate(features, likelihoods, batch_size=self.mini_batch_size, verbose=2)

    def evaluate_generator(self, test_generator, test_steps):
        return self.model.evaluate_generator(generator=test_generator, steps=test_steps, verbose=1)

def load_npz_data(data_file, shuffle = False, num_samples = -1):
    # loaded_array = np.load('./data_v/npz/img-PHOC_test_nopad.npz')
    # loaded_array = np.load('./1220_vu/img-PHOC_train.npz')
    loaded_array = np.load(data_file)

    np.set_printoptions(threshold=np.inf)

    images = loaded_array['array_1']

    print(np.array(images).shape)

    images = preprocess_input(np.array(images, dtype=float))

    images = np.array(images).reshape((-1, 100, 500, 1))
    # images = np.array(images).reshape((-1, 100, 200, 1))

    labels = loaded_array['array_2']
    print(np.array(images).shape)
    print(np.array(labels).shape)

    # random shuffle
    if shuffle:
        rand_index = np.arange(len(labels))
        np.random.shuffle(rand_index)

        images = images[rand_index]
        labels = labels[rand_index]

    if num_samples > 0:
        images = images[:num_samples]
        labels = labels[:num_samples]

    return images, labels

def predict_label(prob, fname):
    vocab_file = '.\data_v\Annotations\CROHME_all\\vocab_all_fixed.txt'

    vocab = [line.strip() for line in open(vocab_file).readlines()]
    vocab = np.array(vocab)

    # print (vocab)
    print (prob.shape)
    result = []
    for p in prob:
        result.append(' '.join(vocab[np.where(p > 0.4)[0]]))

    with open(fname, 'w') as f:
        f.write('\n'.join(result))

def get_heatmap(prob_img, image_size):
    cam = np.array(prob_img).astype(np.float)
    cam = np.clip(cam, 0, 1)
    # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    # resize image
    cam = cv2.resize(cam, image_size)

    cam = cam * 255
    cam = cam.astype('uint8')
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_HOT)

    return heatmap

def visualize_all(image, out_image, sorted_index, vocab):
    heatmap = get_heatmap(np.max(out_image, axis=2), (image.shape[2] * 4, image.shape[1]))

    input_img = np.concatenate(([np.squeeze(image, 0)] * 4), axis=1)

    # input_img = cv2.cvtColor(np.squeeze(input_img, axis=2), cv2.COLOR_GRAY2RGB)

    input_img = input_img * 255
    input_img = input_img.astype('uint8')
    input_img = cv2.applyColorMap(input_img, cv2.COLORMAP_HOT)

    heatmap = cv2.addWeighted(heatmap, 1.0, input_img, 0.3, 0)

    cv2.imshow('input', input_img)

    cv2.imshow('heat map all', heatmap)
    cv2.imwrite('heatmap_all.png', heatmap)

    # heat map
    for class_id in sorted_index:
        print(vocab[class_id])
        heatmap = get_heatmap(out_image[:, :, class_id], (image.shape[2] * 4, image.shape[1]))
        heatmap = cv2.addWeighted(heatmap, 1.0, input_img, 0.3, 0)
        heatmap = cv2.putText(heatmap,
                              '{}: {:.2f}'.format(vocab[class_id].replace('\\', ''), np.max(out_image[:, :, class_id])),
                              (10, 30),
                              cv2.FONT_HERSHEY_DUPLEX,
                              1,
                              (255, 255, 255),
                              2)
        cv2.imshow('heat map', heatmap)
        cv2.imwrite('heatmap_{}.png'.format(vocab[class_id].replace('\\', '')), heatmap)

        cv2.waitKey()

def visualize(image, out_image, sorted_index, vocab):
    heatmap = get_heatmap(np.max(out_image, axis=2), (image.shape[2], image.shape[1]))

    input_img = np.squeeze(image, 0)

    # input_img = cv2.cvtColor(np.squeeze(input_img, axis=2), cv2.COLOR_GRAY2RGB)

    input_img = input_img * 255
    input_img = input_img.astype('uint8')
    input_img = cv2.applyColorMap(input_img, cv2.COLORMAP_HOT)

    heatmap = cv2.addWeighted(heatmap, 1.0, input_img, 0.3, 0)

    cv2.imshow('input', input_img)
    cv2.imwrite('input.png', input_img)

    cv2.imshow('heat map all', heatmap)
    cv2.imwrite('heatmap_all.png', heatmap)

    # cv2.waitKey()

    # heat map
    for class_id in sorted_index:
        print(vocab[class_id])
        heatmap = get_heatmap(out_image[:, :, class_id], (image.shape[2], image.shape[1]))
        heatmap = cv2.addWeighted(heatmap, 1.0, input_img, 0.3, 0)
        heatmap = cv2.putText(heatmap,
                              '{}: {:.2f}'.format(vocab[class_id].replace('\\', ''), np.max(out_image[:, :, class_id])),
                              (10, 30),
                              cv2.FONT_HERSHEY_DUPLEX,
                              0.8,
                              (255, 255, 255),
                              2)
        cv2.imshow('heat map', heatmap)
        cv2.imwrite('heatmap_{}.png'.format(vocab[class_id].replace('\\', '')), heatmap)

        cv2.waitKey()


if __name__ == '__main__':

    initial_epoch = 0 # Khuong 73  #Sasaki4 76 #Sasaki3 76 #Sasaki2 78 #Sasaki1 78 #Sasaki0 76 #CROHME: 67 # attn 69
    # 311 # 320
    mini_batch_size = 1 # 1 # 30
    train = True

    cross_idx = 4

    # prepare and save data
    # prepare_data()

    # load training data

    cnn = CNN_build_model()

    # load weights
    load_prefix = cnn.model.name + '_CROHME_fine_phoclvl1'
    # load_prefix = cnn.model.name + '_CROHME_pretrain_phoclvl1'
    load_prefix = cnn.model.name + '_CROHME_phoclvl1'
    load_prefix = cnn.model.name + '_CROHME_pretrain_finetune'
    load_prefix = cnn.model.name + '_CROHME_2016'
    # load_prefix = cnn.model.name + '_CROHME_2016_Khuong_{}'.format(cross_idx)
    # load_prefix = cnn.model.name + '_CROHME_2016_Sasaki_{}'.format(cross_idx)



    if initial_epoch > 0:
        cnn.model.load_weights(MODEL_PATH + load_prefix + '.ep{:03d}.h5'.format(initial_epoch), by_name=True)
        print ('Model loaded')
        # cnn.model.save_weights(MODEL_PATH + load_prefix + '.ep{:03d}.h5'.format(initial_epoch))
        # print ('Model saved')

    ## training
    if train:
        train_gen, train_len = prepare_data_train(batch_size=mini_batch_size)
        val_gen, val_len = prepare_data_valid(batch_size=mini_batch_size)

        # train_gen, train_len = prepare_data_train_cross(batch_size=mini_batch_size, cross_id=cross_idx)
        # val_gen, val_len = prepare_data_valid_cross(batch_size=mini_batch_size, cross_id=cross_idx)
        print (train_len, val_len)

        # for input, target in val_gen:
        #     print(input.shape, target.shape)

        # cnn.fit(images, labels, initial_epoch)
        cnn.fit_generator(train_generator=train_gen, steps_per_epoch=train_len//mini_batch_size, validation_generator=val_gen,
                      validation_steps=val_len//mini_batch_size, initial_epoch=initial_epoch)

    ## evaluation
    else:
        mini_batch_size = 1

        train_gen, train_len = prepare_data_train(batch_size=mini_batch_size)
        test_gen, test_len, test_data = prepare_data_test(batch_size=mini_batch_size)

        # test_gen, test_len, test_data = prepare_data_test_cross(batch_size=mini_batch_size, cross_id=cross_idx)
        print (test_len)
        test_steps = test_len // mini_batch_size

        # ---------------------------- feature extraction
        # size_file, file, _ = list(zip(*test_data))
        #
        # with open("testdata_file_list.txt", 'w') as f:
        #     f.write('\n'.join(file))
        #
        # predict = cnn.model.predict_generator(test_gen, test_steps, verbose=1)
        # print (predict.shape)
        # np.savetxt("predict.csv", predict, delimiter=",")

        # target = np.concatenate([label for _, label in test_gen], axis=0)
        # print (target.shape)
        # np.savetxt("target.csv", target, delimiter=",")


        # ---------------------------- testing
        # print (cnn.model.evaluate_generator(generator=test_gen, steps=test_steps, verbose=1))

        # ---------------------------- visualize

        vocab_path = 'vocab_syms.txt'
        vocab = [line.strip() for line in open(vocab_path).readlines()]

        labels = []
        index = 0
        for image, label in train_gen:
            print(label[0])

            cv2.imshow('input_img', np.squeeze(image, axis=0))
            cv2.imwrite('input_img.png', np.squeeze(image, axis=0))
            out_image = cnn.model.predict(image)
            out_image = np.squeeze(out_image, axis=0)
            sorted_index = np.argsort(label[0])[::-1][:12]

            visualize(image, out_image, sorted_index, vocab)

        # ---------------------------- classification check

        # vocab_path = 'vocab_syms.txt'
        # vocab = [line.strip() for line in open(vocab_path).readlines()]
        #
        # labels = []
        # index = 0
        # for image, label in train_gen:
        #
        #     cv2.imshow('input_img', np.squeeze(image, axis=0))
        #     # print (image.shape)
        #     # cv2.imwrite('input_img.png', np.squeeze(image, axis=0))
        #     pred = cnn.model.predict(image)
        #     # print(pred.shape)
        #     # out_image = np.squeeze(out_image, axis=0)
        #     sorted_index = np.argsort(label[0])[::-1][:12]
        #     print('\t'.join([vocab[index] for index in sorted_index]))
        #     print(np.round(pred[0][0][sorted_index], 1))
        #     print(np.round(pred[1][0][sorted_index], 1))
        #
        #     print(label[0][sorted_index])
        #
        #     cv2.waitKey()

