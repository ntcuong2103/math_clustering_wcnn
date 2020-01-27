from keras.layers import GlobalAveragePooling2D, Average, Add, TimeDistributed, Maximum, Lambda, Conv2D, AveragePooling2D, MaxPooling2D, Input, Concatenate, Dense, Dropout, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model
from keras import backend as K
from spp.SpatialPooling2D import SpatialPooling2D
from spp.SpatialPyramidPooling2D import SpatialPyramidPooling
import tensorflow as tf

def buildCNN2Lvl(input_shape, global_pool = False, single_level = False):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape)(inputs)
    # x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 32 / 2
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 32 / 4
    x_lvl1 = Conv2D(512, (8, 8), padding='same', activation='relu')(x)
    x_lvl2 = Conv2D(512, (3, 3), padding='same', activation='relu')(x_lvl1)


    if global_pool:
        x_lvl1 = GlobalMaxPooling2D()(x_lvl1)
        x_lvl2 = GlobalMaxPooling2D()(x_lvl2)

    if single_level:
        x_merge = x_lvl1
    else:
        x_merge = Concatenate()([x_lvl1, x_lvl2])

    model = Model(inputs, x_merge)
    model.summary()

    return model

def multiScaleCNNFeature(inputs, input_shape):
    inputs_s2 = AveragePooling2D()(inputs)
    inputs_s3 = AveragePooling2D()(inputs_s2)

    modelCNN2Lvl = buildCNN2Lvl(input_shape, global_pool=False, single_level=False)

    # for layer in modelCNN2Lvl.layers:
    #     layer.trainable = False

    out_1 = modelCNN2Lvl(inputs)
    out_2 = modelCNN2Lvl(inputs_s2)
    out_3 = modelCNN2Lvl(inputs_s3)

    return [out_1, out_2, out_3]

def attentive_sum_multiplication_v1(feature, attention):
    # feature: b, n_class, h, w, n_feature
    # attention: b, h, w, n_class
    # softmax normalization
    # attention = tf.exp(attention) / tf.reduce_sum(tf.exp(attention), axis=(1,2), keepdims=True)


    attention_onehot = attention >= tf.reduce_max(attention, axis=(1, 2), keepdims=True)
    attention_high = attention >= 0.2

    attention = tf.logical_or(attention_onehot, attention_high)
    attention = tf.cast(attention, tf.float32)

    x = tf.expand_dims(tf.transpose(attention, [3, 0, 1, 2]), -1) * feature
    x = tf.transpose(x, [1, 0, 2, 3, 4])
    x = tf.reduce_sum(x, axis=2)
    x = tf.reduce_sum(x, axis=2)
    return x

def attentive_sum_multiplication(feature, attention):
    # feature: b, n_class, h, w, n_feature
    # attention: b, h, w, n_class
    # softmax normalization
    # attention = tf.exp(attention) / tf.reduce_sum(tf.exp(attention), axis=(1,2), keepdims=True)


    # attention_onehot = attention >= tf.reduce_max(attention, axis=(1, 2), keepdims=True)
    # attention = tf.maximum(tf.cast(attention_onehot, tf.float32), attention)

    divident = 1e-10 + tf.expand_dims(tf.reduce_sum(attention, axis=(1,2)), 2)

    x = tf.expand_dims(tf.transpose(attention, [3, 0, 1, 2]), -1) * feature
    x = tf.transpose(x, [1, 0, 2, 3, 4])
    x = tf.reduce_sum(x, axis=(2,3))

    x = x / divident
    return x


def buildClassifier(n_out, input_shape=[None, None, 1024]):
    inputs = Input(shape=input_shape)
    dense_1 = Dense(1024, activation='relu')
    dense_2 = Dense(n_out, activation='sigmoid')

    fc = dense_1(inputs)
    fc = Dropout(0.5)(fc)
    outputs = dense_2(fc)

    return Model(inputs, outputs)

def randomSelect(input1, input2, threshold = 0.5):
    sel = K.cast(K.random_uniform([1]) > threshold, K.floatx())
    return sel * input1 + (1 - sel) * input2

def buildLocalAttentiveModel(input_shape, n_out, pool='max', attentive=True):
    inputs = Input(shape=input_shape)

    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)
    classifier = buildClassifier(n_out)

    out_a1 = classifier(out_1)
    out_a2 = classifier(out_2)
    out_a3 = classifier(out_3)


    if pool == 'max':
        # global max pooling
        out_g1_cls = GlobalMaxPooling2D()(out_a1)
        out_g2_cls = GlobalMaxPooling2D()(out_a2)
        out_g3_cls = GlobalMaxPooling2D()(out_a3)
    else:
        # global average pooling

        out_g1 = GlobalAveragePooling2D()(out_1)
        out_g2 = GlobalAveragePooling2D()(out_2)
        out_g3 = GlobalAveragePooling2D()(out_3)
        out_g1_cls = classifier(out_g1)
        out_g2_cls = classifier(out_g2)
        out_g3_cls = classifier(out_g3)

    # outputs_gp = Maximum()([out_g1_cls, out_g2_cls, out_g3_cls])
    outputs_gp = Maximum()([out_g1_cls, out_g2_cls])

    if attentive:

        out_1 = Lambda(lambda feature: attentive_sum_multiplication(feature[0], feature[1]))([out_1, out_a1])
        out_2 = Lambda(lambda feature: attentive_sum_multiplication(feature[0], feature[1]))([out_2, out_a2])
        out_3 = Lambda(lambda feature: attentive_sum_multiplication(feature[0], feature[1]))([out_3, out_a3])


        # b, n_class, n_feature
        out_1_cls = classifier(out_1)
        out_1_cls = Lambda(lambda out: tf.matrix_diag_part(out))(out_1_cls)

        out_2_cls = classifier(out_2)
        out_2_cls = Lambda(lambda out: tf.matrix_diag_part(out))(out_2_cls)

        out_3_cls = classifier(out_3)
        out_3_cls = Lambda(lambda out: tf.matrix_diag_part(out))(out_3_cls)

        # outputs_att = Maximum()([out_1_cls, out_2_cls, out_3_cls])
        outputs_att = Maximum()([out_1_cls, out_2_cls])

        # outputs = Average()([outputs_att, outputs_gp])

        # outputs = [outputs_att, outputs_gp]

        outputs = Lambda(lambda out: randomSelect(out[0], out[1], 0.5)) ([outputs_att, outputs_gp])
    else:
        outputs = outputs_gp

    model = Model(inputs, outputs, name='cnn_multiscale_2lvl_ft2_gatp_gmp')
    model.summary()
    return model

# For debug
def buildLocalAttentiveModelAll(input_shape, n_out):
    inputs = Input(shape=input_shape)

    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)
    classifier = buildClassifier(n_out)

    out_g1 = GlobalMaxPooling2D()(out_1)
    out_g2 = GlobalMaxPooling2D()(out_2)
    out_g3 = GlobalMaxPooling2D()(out_3)

    out_a1 = classifier(out_1)
    out_a2 = classifier(out_2)
    out_a3 = classifier(out_3)

    out_1 = Lambda(lambda feature: attentive_sum_multiplication(feature[0], feature[1]))([out_1, out_a1])
    out_2 = Lambda(lambda feature: attentive_sum_multiplication(feature[0], feature[1]))([out_2, out_a2])
    out_3 = Lambda(lambda feature: attentive_sum_multiplication(feature[0], feature[1]))([out_3, out_a3])


    # b, n_class, n_feature
    # out_sum_att = Maximum()([out_1, out_2, out_3])
    # outputs_att = classifier(out_sum_att)
    # outputs_att = Lambda(lambda out: tf.matrix_diag_part(out), name='outputs_att')(outputs_att)

    out_1_cls = classifier(out_1)
    out_1_cls = Lambda(lambda out: tf.matrix_diag_part(out))(out_1_cls)

    out_2_cls = classifier(out_2)
    out_2_cls = Lambda(lambda out: tf.matrix_diag_part(out))(out_2_cls)

    out_3_cls = classifier(out_3)
    out_3_cls = Lambda(lambda out: tf.matrix_diag_part(out))(out_3_cls)

    outputs_att = Maximum()([out_1_cls, out_2_cls, out_3_cls])


    out_sum_gp = Maximum()([out_g1, out_g2, out_g3])
    outputs_gp = classifier(out_sum_gp)

    out_g1_cls = classifier(out_g1)
    out_g2_cls = classifier(out_g2)
    out_g3_cls = classifier(out_g3)


    outputs = Concatenate()([out_1_cls, out_2_cls, out_3_cls, outputs_att, out_g1_cls, out_g2_cls, out_g3_cls, outputs_gp])

    model = Model(inputs, outputs, name='cnn_multiscale_3lvl_ft2_diag_gp_multiloss')
    model.summary()
    return model

# Visualization
def buildVisualizeModel(input_shape, n_out, out_size = (5, 10)): # out_size (1, 1), (3,5)
    # out_size = (tf.constant(50), tf.constant(100))
    # out_size = (5, 10)
    inputs = Input(shape=input_shape)

    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)

    classifier = buildClassifier(n_out)

    out_1 = classifier(out_1)
    out_2 = classifier(out_2)
    out_3 = classifier(out_3)

    # resize all output images
    out_1 = SpatialPooling2D(out_size)(out_1)
    out_2 = SpatialPooling2D(out_size)(out_2)
    out_3 = SpatialPooling2D(out_size)(out_3)

    # resize all output images
    # out_1 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_1)
    # out_2 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_2)
    # out_3 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_3)

    outputs = Maximum()([out_1, out_2, out_3])

    model = Model(inputs, outputs, name='cnn_multiscale_3lvl_ft2_diag_gp_multiloss')
    return model

def buildVisualizeModel2(input_shape, n_out):
    out_size = (tf.constant(50), tf.constant(100))
    inputs = Input(shape=input_shape)
    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)

    classifier = buildClassifier(n_out)

    out_1 = classifier(out_1)
    out_2 = classifier(out_2)
    out_3 = classifier(out_3)

    # resize all output images
    out_1 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_1)
    out_2 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_2)
    out_3 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_3)

    outputs = Maximum()([out_1, out_2, out_3])
    # outputs = out_3

    model = Model(inputs, outputs)
    return model

def buildVisualizeModelAll(input_shape, n_out):
    out_size = (tf.constant(50), tf.constant(100))
    inputs = Input(shape=input_shape)
    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)

    classifier = buildClassifier(n_out)

    out_1 = classifier(out_1)
    out_2 = classifier(out_2)
    out_3 = classifier(out_3)

    # resize all output images
    out_1 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_1)
    out_2 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_2)
    out_3 = Lambda(lambda image: tf.image.resize_images(image, out_size))(out_3)

    out_max = Maximum()([out_1, out_2, out_3])

    outputs = Concatenate(axis=2)([out_max, out_1, out_2, out_3])
    # outputs = out_3

    model = Model(inputs, [outputs], name='cnn_multiscale_3lvl_ft2_gattp_gmp')
    return model

def buildVisualizeModelAll2(input_shape, n_out):
    inputs = Input(shape=input_shape)
    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)

    classifier = buildClassifier(n_out)

    out_1 = classifier(out_1)
    out_2 = classifier(out_2)
    out_3 = classifier(out_3)

    model = Model(inputs, [out_1, out_2, out_3], name='cnn_multiscale_3lvl_ft2_gattp_gmp')
    return model

# Clustering features
def buildSpatialFeatureModel(input_shape, n_out, out_list = [(1,1),(3,5),(3,7),(5,7)]): # out_size (1, 1), (3,7)
    # out_size = (tf.constant(50), tf.constant(100))
    # out_size = (5, 10)
    inputs = Input(shape=input_shape)

    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)

    classifier = buildClassifier(n_out)

    out_1 = classifier(out_1)
    out_2 = classifier(out_2)
    out_3 = classifier(out_3)


    # resize all output images
    out_1 = SpatialPyramidPooling(out_list)(out_1)
    out_2 = SpatialPyramidPooling(out_list)(out_2)
    out_3 = SpatialPyramidPooling(out_list)(out_3)

    outputs = Maximum()([out_1, out_2, out_3])
    # outputs = Maximum()([out_1, out_2])
    # outputs = out_1

    model = Model(inputs, outputs, name='cnn_multiscale_3lvl_ft2_gattp_gmp')
    return model

# Classification
def buildClassificationModel(input_shape, n_out): # out_size (1, 1), (3,7)
    inputs = Input(shape=input_shape)

    out_1, out_2, out_3 = multiScaleCNNFeature(inputs, input_shape)

    classifier = buildClassifier( n_out)

    out_g1 = GlobalMaxPooling2D()(out_1)
    out_g2 = GlobalMaxPooling2D()(out_2)
    out_g3 = GlobalMaxPooling2D()(out_3)

    out_1 = classifier(out_g1)
    out_2 = classifier(out_g2)
    out_3 = classifier(out_g3)

    outputs = Maximum()([out_1, out_2, out_3])

    model = Model(inputs, outputs, name='cnn_multiscale_3lvl_ft2_diag_gp_multiloss')
    return model

# Pretrain
def buildPretrainModel(input_shape, n_out):
    # shape (b, n_group (None), h, w, 1)
    inputs = Input(shape=[None] + input_shape)

    modelCNN2Lvl = buildCNN2Lvl(input_shape, global_pool=True)


    features = TimeDistributed(modelCNN2Lvl, name='model_1')(inputs)
    features = Lambda(lambda x: K.max(x, axis=1))(features)

    classifier = buildClassifier(n_out)


    outputs = classifier(features)

    model = Model(inputs, outputs, name='cnn_multiscale_3lvl_ft2_diag_gp_multiloss')

    model.summary()

    return  model

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model = buildLocalAttentiveModelAll([None, None, 1], 97)
    for layer in model.layers:
        print (layer.name)
        symbolic_weights = layer.weights
        weight_names = []
        for i, (w) in enumerate(symbolic_weights):
            if hasattr(w, 'name') and w.name:
                name = str(w.name)
            else:
                name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))

        print (weight_names)
    model.save_weights('test.h5')
    print ('model saved')

    # model2 = buildVisualizeModel([None, None, 1], 97)
    # for layer in model2.layers:
    #     print (layer.name)
    #     symbolic_weights = layer.weights
    #     weight_names = []
    #     for i, (w) in enumerate(symbolic_weights):
    #         if hasattr(w, 'name') and w.name:
    #             name = str(w.name)
    #         else:
    #             name = 'param_' + str(i)
    #         weight_names.append(name.encode('utf8'))
    #
    #     print (weight_names)
    #
    # model2.load_weights('test.h5', by_name=True)
    # print ('model loaded')