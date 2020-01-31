"""
2019/01
@author　Ushizawa
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from datetime import datetime
from CNN_multiscale_3lvl_2ft_pretrain_diag_viz_multiloss import buildVisualizeModel2 as buildModel
from CNN_multiscale_3lvl_2ft_pretrain_diag_viz_multiloss import buildVisualizeModelAll2 as buildModel
# from CNN_multiscale_3lvl_2ft_pretrain_diag_viz_multiloss import buildSpatialFeatureModel as buildModel

from prepare_data_synthesis import prepare_data_train, prepare_data_valid, prepare_data_test
# from prepare_data_synthesis import prepare_data_train_cross, prepare_data_valid_cross, prepare_data_test_cross

from config import CONFIGURATION, DATASET

MODEL_PATH = CONFIGURATION["MODEL_PATH"]
LOG_PATH = CONFIGURATION["LOG_PATH"]

os.environ['CUDA_VISIBLE_DEVICES'] = CONFIGURATION["CUDA_DEVICE"]

def gpu_config():
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

class CNN_build_model():
    def __init__(self):
        self.num_epoch = 400
        channels = 1
        # n_out = 97
        n_out = 101

        self.model = buildModel(input_shape=[None, None, channels], n_out=n_out)
        self.model.summary()

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
        save_prefix = self.model.name + '_CROHME_2016_Khuong_0'

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

    def evaluate(self, features, likelihoods):
        return self.model.evaluate(features, likelihoods, batch_size=self.mini_batch_size, verbose=2)

    def evaluate_generator(self, test_generator, test_steps):
        return self.model.evaluate_generator(generator=test_generator, steps=test_steps, verbose=1)

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

def visualize(prefix, image, out_image, sorted_index, vocab, visualize_class = False):
    heatmap = get_heatmap(np.max(out_image, axis=2), (image.shape[1], image.shape[0]))

    input_img = image
    # input_img = np.squeeze(image, 0)


    input_img = input_img * 255
    input_img = input_img.astype('uint8')
    # input_img = cv2.applyColorMap(input_img, cv2.COLORMAP_HOT)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)

    print(input_img.shape, heatmap.shape)
    heatmap = cv2.addWeighted(heatmap, 0.6, input_img, 0.4, 0)

    out_image_path = 'heatmap_all_level'
    os.makedirs(out_image_path, exist_ok=True)
    heatmap_fn = os.path.join(out_image_path, prefix + '.png')
    cv2.imwrite(heatmap_fn, heatmap)

    if not visualize_class:
        return
    # heat map
    for class_id in sorted_index:
        print(vocab[class_id])
        heatmap = get_heatmap(out_image[:, :, class_id], (image.shape[1], image.shape[0]))
        heatmap = cv2.addWeighted(heatmap, 0.6, input_img, 0.4, 0)
        # np.max(out_image[:, :, class_id])
        heatmap = cv2.putText(heatmap,
                              '{}'.format(vocab[class_id].replace('\\', '')),
                              (10, 30),
                              cv2.FONT_HERSHEY_DUPLEX,
                              0.8,
                              (255, 255, 255),
                              2)
        cv2.imwrite(os.path.join(out_image_path, prefix + '_{}.png'.format(class_id)), heatmap)


if __name__ == '__main__':
    initial_epoch = CONFIGURATION["INITIAL_EPOCH"]
    mini_batch_size = CONFIGURATION["BATCH_SIZE"] #5 # 30
    train = CONFIGURATION["IS_TRAINED"]

    cross_idx = 4

    gpu_config()
    cnn = CNN_build_model()

    # load weights
    load_prefix = CONFIGURATION["MODEL_NAME"] + '_CROHME_' + DATASET

    if initial_epoch > 0:
        cnn.model.load_weights(MODEL_PATH + load_prefix + '.ep{:03d}.h5'.format(initial_epoch), by_name=True)
        print ('Model loaded:{}'.format(MODEL_PATH + load_prefix + '.ep{:03d}.h5'.format(initial_epoch)))

    ## training
    if train:
        pass
    ## evaluation
    else:
        # train_gen, train_len = prepare_data_train(batch_size=mini_batch_size)
        test_gen, test_len, test_data = prepare_data_test(batch_size=mini_batch_size)
        # print(test_data[:10])

        # test_gen, test_len, test_data = prepare_data_test_cross(batch_size=mini_batch_size, cross_id=cross_idx)
        print (test_len)
        test_steps = test_len // mini_batch_size

        # ---------------------------- feature extraction

        # output_path = 'clustering_experiment/khuong_dmix/attn'
        # os.makedirs(output_path, exist_ok=True)
        #
        # size_file, file, _ = list(zip(*test_data))
        #
        # with open(os.path.join(output_path, "testdata_file_list.txt"), 'w') as f:
        #     f.write('\n'.join(file))
        #
        # predict = cnn.model.predict_generator(test_gen, test_steps, verbose=1)
        # print (predict.shape)
        # np.savetxt(os.path.join(output_path, "predict.csv"), predict, delimiter=",")
        #
        # # target = np.concatenate([label for _, label in test_gen], axis=0)
        # # print (target.shape)
        # # np.savetxt(os.path.join(output_path, "target.csv"), target, delimiter=",")
        # exit(0)


        # ---------------------------- testing
        # print (cnn.model.evaluate_generator(generator=test_gen, steps=test_steps, verbose=1))

        # ---------------------------- visualize

        vocab_path = 'vocab_syms.txt'
        vocab = [line.strip() for line in open(vocab_path).readlines()]

        labels = []
        index = 0
        for images, labels in test_gen:
            # print(label[0])
            # cv2.imshow('input_img', np.squeeze(image, axis=0))
            # cv2.imwrite('input_img.png', np.squeeze(image, axis=0))
            out_images_multi = cnn.model.predict(images)

            for lvl, out_images in enumerate(out_images_multi):
                # print(out_image[0].shape)
                index_in = index
                for image, out_image, label in zip(images, out_images, labels):
                    print('image.shape, out_image.shape: ', image.shape, out_image.shape)
                    # out_image = np.squeeze(out_image, axis=0)
                    sorted_index = np.argsort(label)[::-1][:12]
                    print(sorted_index)
                    prefix = os.path.basename(test_data[index_in][1]).split('.')[0]

                    visualize(prefix + '_l{}'.format(lvl), image, out_image, sorted_index, vocab)
                    index_in += 1

            index = index_in

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

