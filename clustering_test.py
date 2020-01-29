
import numpy as np
import scipy.spatial.distance as dis
import csv
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
# from sklearn.cluster import MiniBatchKMeans as KMeans
import collections
import os

def Kmeans_cluster (D, k):
    D = np.array(D, dtype=np.float32)
    pred = MiniBatchKMeans(n_clusters=k, verbose=False).fit_predict(D)
    pred.tolist()
    C = {i: np.array([index for index, label in enumerate(pred) if label == i]) for i in range(0, k)}
    return C

def Agg_cluster (D, k):
    D = np.array(D, dtype=np.float32)
    pred = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average').fit_predict(D)
    pred.tolist()
    C = {i: np.array([index for index, label in enumerate(pred) if label == i]) for i in range(0, k)}
    return C


def my_manhattan(point1, point2):
    dimension = len(point1)
    result = 0.0
    for i in range(dimension):
        result += abs(point1[i] - point2[i]) * 0.1
    return result

def Kmedoids_cluster(D, k):
    from pyclustering.cluster.kmedoids import kmedoids, distance_metric
    from pyclustering.utils.metric import type_metric
    from scipy.spatial.distance import cosine
    # metric = distance_metric(type_metric.EUCLIDEAN)
    metric = distance_metric(type_metric.USER_DEFINED, func=hierachical_distance)

    # EUCLIDEAN = 0
    #
    # EUCLIDEAN_SQUARE = 1
    #
    # MANHATTAN = 2
    #
    # CHEBYSHEV = 3
    #
    # MINKOWSKI = 4
    #
    # CANBERRA = 5
    #
    # CHI_SQUARE = 6
    #
    # USER_DEFINED = 1000

    initial_medoids = np.random.choice(np.arange(len(D)), k)
    # print(initial_medoids)
    # print(D.shape)
    kmedoids_instance = kmedoids(D, initial_medoids, metric=metric)

    kmedoids_instance.process()

    pred = kmedoids_instance.get_clusters()
    # print (pred)


    # pred.tolist()
    C = {i: np.array(c) for i, c in enumerate(pred)}
    # C = {i: np.array([index for index, label in enumerate(pred) if label == i]) for i in range(0, k)}
    return C

def clustering_result_check (test_lst, C, path):
    img_id = [i.split(' ')[0] for i in test_lst]        # file.png
    img_label = [i.split(' ')[-1] for i in test_lst]    # 正解クラスタ

    with open(path, 'w') as f:
        for cluster in C.keys():
            for i in C[cluster]:
                if i == len(test_lst):  # 追加した人工medoid だったら
                    continue
                f.writelines('{}\t{}\t{}\t{}\n'.format(cluster, img_label[i], i, img_id[i]))

def calc_purity(result_path, purity_path):

    # purity を計算する場合
    with open(result_path, 'r') as f:
        data = [i.strip().split('\t') for i in f]
        # data = [list(map(int, i)) for i in data]
        # data_cluster = [int(i[0]) for i in data]
        data_cluster, group_ids, _, _ = list(zip(*data))

    with open(purity_path, 'a') as f:
        major = total = 0
        for num in list(set(data_cluster)):
            # 同じクラスタのdata
            data_tmp = [_group for _cluster, _group in zip(data_cluster, group_ids) if _cluster == num]
            item_num = collections.Counter(data_tmp).most_common()
            # purity の計算
            purity = item_num[0][1] / sum([i[1] for i in item_num])
            write_data = str(num)+'\t'+str(item_num)+'\t'+str(purity)+ '\t' + \
                         str(item_num[0][1]) + '\t' + str(sum([i[1] for i in item_num])) + '\n'
            f.writelines(write_data)
            major += item_num[0][1]
            total += sum([i[1] for i in item_num])
            # print(write_data)

        return major, total

def hierachical_distance(X1, X2):
    return 1.0 - np.sum(X1 * X2) / np.sum(np.maximum(X1, X2))

def get_dirname_filename(filename):
    return os.path.join(os.path.basename(os.path.dirname(filename)),
                        os.path.basename(filename).split('.')[0])

def clustering_CROHME():
    test_folder = "clustering_experiment/crohme2019/gap"
    # cluster_num = 36
    cluster_num = 49 # 2019

    predict_csv_file = os.path.join(test_folder, "predict.csv")
    test_filelist = os.path.join(test_folder, "testdata_file_list.txt")

    os.makedirs(os.path.join(test_folder, "kmeans"), exist_ok=True)

    test_files = [line.strip() for line in open(test_filelist).readlines()]
    # test_groups = [line.strip().split('\\')[-1].split('_')[0] for line in open(test_filelist).readlines()]

    fn_to_group = {line.strip().split('\t')[0]:'_'.join(sorted(list(set(line.strip().split('\t')[1].split())))) for line in open('crohme2019_test_annotation_sym_seq.txt').readlines()}
    print(fn_to_group)
    test_groups = [fn_to_group[get_dirname_filename(fn)] for fn in test_files]

    print(test_groups)


    print ((list(set(test_groups))))

    # print (test_groups, test_files)

    spatial_sizes = [(1,1),(3,5),(3,7),(5,7)]
    num_classes = 101
    y_pred = np.array(np.loadtxt(predict_csv_file, delimiter=",")).astype(float)
    print(y_pred.shape)

    predictions = []
    begin = 0
    for spatial_size in spatial_sizes:
        feature_len = spatial_size[0] * spatial_size[1] * num_classes
        predictions.append(y_pred[:, begin:begin+feature_len])
        print(begin, feature_len)
        begin += feature_len

    combine_preds = []
    for i in range (1, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[i]), axis=-1))

    for i in range (2, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[i]), axis=-1))

    for i in range (3, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[2], predictions[i]), axis=-1))

    predictions += combine_preds

    # compute pairwise features
    from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

    # - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
    #   'manhattan']. These metrics support sparse matrix inputs.
    #
    # - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
    #   'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
    #   'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
    #   'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    #   See the documentation for scipy.spatial.distance for details on these
    #   metrics. These metrics do not support sparse matrix inputs.

    # cosine_features = []
    # for f_id in range(len(predictions)):
    #     # cosine_features.append(pairwise_distances(predictions[f_id], metric=hierachical_distance)) #
    #     cosine_features.append(1 - cosine_similarity(predictions[f_id]))
    # predictions = cosine_features


    purity = []
    loop = 5
    for _ in range(loop):
        print ('loop')
        for f_id in range(len(predictions)):

            result_file = os.path.join(test_folder, 'kmeans', 'kmeans_f{}_{}.txt'.format(f_id, cluster_num))
            purity_file = os.path.join(test_folder, 'kmeans', 'purity_f{}_{}.txt'.format(f_id, cluster_num))

            answer_ids, files, features = test_groups, test_files, predictions[f_id]
            # clustering answers

            features = 1 - cosine_similarity(features)
            # C = Agg_cluster(features, cluster_num)
            C = Kmeans_cluster(features, cluster_num)

            # C = Kmedoids_cluster(features, cluster_num)

            with open(result_file, 'w') as f:
                for cluster in C.keys():
                    for i in C[cluster]:
                        f.writelines('{}\t{}\t{}\t{}\n'.format(cluster, answer_ids[i], i, files[i]))

            major, total = calc_purity(result_file, purity_file)

            purity.append(float(major) / total)

    purity = np.array(purity).reshape(loop, len(predictions))
    print(np.average(purity, axis=0), np.var(purity, axis=0))
    return np.average(purity, axis=0), np.var(purity, axis=0)


def clustering_Sasaki(cross_idx):
    # test_folder = "./clustering_experiment/sasaki/attn_retrain/{}/".format(cross_idx)
    test_folder = './clustering_experiment/sasaki/attn/'
    cluster_num = 50

    predict_csv_file = test_folder + "predict.csv"
    test_filelist = test_folder + "testdata_file_list.txt"

    os.makedirs(test_folder + "kmeans", exist_ok=True)


    test_filelist = [(line.strip().split('\\')[-1].split('_')[0], line.strip()) for line in open(test_filelist).readlines()]
    test_groups, test_files = list(zip(*test_filelist))

    spatial_sizes = [(1,1),(3,5),(3,7),(5,7)]
    num_classes = 101
    y_pred = np.array(np.loadtxt(predict_csv_file, delimiter=",")).astype(float)
    # print(y_pred.shape)

    predictions = []
    begin = 0
    for spatial_size in spatial_sizes:
        feature_len = spatial_size[0] * spatial_size[1] * num_classes
        predictions.append(y_pred[:, begin:begin+feature_len])
        # print(begin, feature_len)
        begin += feature_len

    combine_preds = []
    for i in range (1, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[i]), axis=-1))

    for i in range (2, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[i]), axis=-1))

    for i in range (3, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[2], predictions[i]), axis=-1))

    predictions += combine_preds

    # cosine dissimilarity
    from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

    # cosine_features = []
    # for f_id in range(len(predictions)):
    #     cosine_features.append(pairwise_distances(predictions[f_id], metric='euclidean')) #
    #     # cosine_features.append(1 - cosine_similarity(predictions[f_id]))
    # predictions = cosine_features

    purity = []

    loop = 5

    for _ in range(loop):
        print('clustering ... ')
        for f_id in range(len(predictions)):

            result_file = test_folder + 'kmeans/kmeans_f{}_{}.txt'.format(f_id, cluster_num)
            purity_file = test_folder + 'kmeans/purity_f{}_{}.txt'.format(f_id, cluster_num)

            answer_ids, files, features = test_groups, test_files, predictions[f_id]
            # clustering answers

            # features = 1 - cosine_similarity(features)

            # C = Agg_cluster(features, cluster_num)
            C = Kmeans_cluster(features, cluster_num)

            with open(result_file, 'w') as f:
                for cluster in C.keys():
                    for i in C[cluster]:
                        f.writelines('{}\t{}\t{}\t{}\n'.format(cluster, answer_ids[i], i, files[i]))

            major, total = calc_purity(result_file, purity_file)

            purity.append(float(major) / total)

    purity = np.array(purity).reshape(loop, len(predictions))
    print(np.average(purity, axis=0), np.var(purity, axis=0))
    return np.average(purity, axis=0), np.var(purity, axis=0)


def clustering_Khuong(cross_idx):
    test_folder = "./clustering_experiment/khuong/attn_retrain/{}/".format(cross_idx)
    cluster_num = 3

    # test_set_path = test_folder + "test_set.txt"
    predict_csv_file = test_folder + "predict.csv"
    test_filelist = test_folder + "testdata_file_list.txt"

    os.makedirs(test_folder + "kmeans", exist_ok=True)


    test_filelist = [(line.strip().split('\\')[-1].split('.')[0][:2], line.strip().split('\\')[-1].split('.')[0], line.strip()) for line in open(test_filelist).readlines()]
    questions, test_groups, test_files = list(zip(*test_filelist))

    print(questions, test_groups, test_files)

    print (len(questions))
    print (list(set(test_groups)))

    spatial_sizes = [(1,1),(3,5),(3,7),(5,7)]
    num_classes = 101
    y_pred = np.array(np.loadtxt(predict_csv_file, delimiter=",")).astype(float)
    # y_pred = y_pred[:, :97]
    # print(y_pred.shape)

    predictions = []
    begin = 0
    for spatial_size in spatial_sizes:
        feature_len = spatial_size[0] * spatial_size[1] * num_classes
        predictions.append(y_pred[:, begin:begin+feature_len])
        # print(begin, feature_len)
        begin += feature_len

    combine_preds = []
    for i in range (1, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[i]), axis=-1))

    for i in range (2, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[i]), axis=-1))

    for i in range (3, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[2], predictions[i]), axis=-1))

    predictions += combine_preds


    print (predictions[0].shape, predictions[-1].shape)

    # cosine dissimilarity
    from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

    # cosine_features = []
    # for f_id in range(len(predictions)):
    #     cosine_features.append(1 - cosine_similarity(predictions[f_id]))
    # predictions = cosine_features
    #
    # print('finished computed features')


    purity = []
    loop = 5

    for _ in range(loop):
        for f_id in range(len(predictions)):
            major = total = 0

            purity_file = test_folder + 'kmeans/purity_f{}_{}.txt'.format(f_id, cluster_num)

            for cluster_question in list(set(questions)):

                result_file = test_folder + 'kmeans/{}_kmeans_f{}_{}.txt'.format(cluster_question, f_id, cluster_num)

                answers = []
                for question, answ, test_file, feature in zip(questions, test_groups, test_files, predictions[f_id]):
                    # print (question, cluster_question)
                    if question == cluster_question:
                        answers.append((answ, test_file, feature))

                # print(cluster_question)
                # print(len(answers))


                answer_ids, files, features = list(zip(*answers))
                # clustering answers
                # features = 1 - cosine_similarity(features)

                C = Kmeans_cluster(features, cluster_num)
                # C = Agg_cluster(features, cluster_num)

                with open(result_file, 'w') as f:
                    for cluster in C.keys():
                        for i in C[cluster]:
                            f.writelines('{}\t{}\t{}\t{}\n'.format(cluster, answer_ids[i], i, files[i]))

                _major, _total = calc_purity(result_file, purity_file)
                major += _major
                total += _total

            # print (float(major)/total)
            purity.append(float(major)/total)

    purity = np.array(purity).reshape(loop, len(predictions))

    print(np.average(purity, axis=0), np.var(purity, axis=0))

    return np.average(purity, axis=0), np.var(purity, axis=0)


def clustering_DsetMix():
    test_folder = 'clustering_experiment/khuong_dmix/attn'
    # cluster_num = 10

    # test_set_path = test_folder + "test_set.txt"
    predict_csv_file = os.path.join(test_folder, "predict.csv")
    test_filelist = os.path.join(test_folder, "testdata_file_list.txt")

    os.makedirs(os.path.join(test_folder, "kmeans"), exist_ok=True)

    test_files = [line.strip() for line in open(test_filelist).readlines()]


    questions, test_groups, test_files = list(zip(*[(fn.split('/')[-3],
                                                     fn.split('/')[-2],
                                                     fn) for fn in test_files]))

    print(questions, test_groups, test_files)

    print (len(questions))
    print (list(set(test_groups)))

    spatial_sizes = [(1,1),(3,5),(3,7),(5,7)]
    num_classes = 101
    y_pred = np.array(np.loadtxt(predict_csv_file, delimiter=",")).astype(float)
    # y_pred = y_pred[:, :97]
    # print(y_pred.shape)

    predictions = []
    begin = 0
    for spatial_size in spatial_sizes:
        feature_len = spatial_size[0] * spatial_size[1] * num_classes
        predictions.append(y_pred[:, begin:begin+feature_len])
        # print(begin, feature_len)
        begin += feature_len

    combine_preds = []
    for i in range (1, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[i]), axis=-1))

    for i in range (2, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[i]), axis=-1))

    for i in range (3, len(predictions)):
        combine_preds.append(np.concatenate((predictions[0], predictions[1], predictions[2], predictions[i]), axis=-1))

    predictions += combine_preds


    print (predictions[0].shape, predictions[-1].shape)

    # cosine dissimilarity
    from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

    # cosine_features = []
    # for f_id in range(len(predictions)):
    #     cosine_features.append(1 - cosine_similarity(predictions[f_id]))
    # predictions = cosine_features
    #
    # print('finished computed features')


    purity = []
    loop = 5

    for _ in range(loop):
        for f_id in range(len(predictions)):
            print(f_id, '----------------------')
            major = total = 0

            purity_file = os.path.join(test_folder,
                                       'kmeans/purity_f{}.txt'.format(f_id))

            for cluster_question in list(set(questions)):

                result_file = os.path.join(test_folder,
                                           'kmeans/{}_kmeans_f{}.txt'.format(cluster_question, f_id))

                answers = []
                for question, answ, test_file, feature in zip(questions, test_groups, test_files, predictions[f_id]):
                    # print (question, cluster_question)
                    if question == cluster_question:
                        answers.append((answ, test_file, feature))


                answer_ids, files, features = list(zip(*answers))
                cluster_num = len(list(set(answer_ids)))
                print(cluster_question, len(answers), cluster_num)


                # clustering answers
                features = 1 - cosine_similarity(features)

                C = Kmeans_cluster(features, cluster_num)
                # C = Agg_cluster(features, cluster_num)

                with open(result_file, 'w') as f:
                    for cluster in C.keys():
                        for i in C[cluster]:
                            f.writelines('{}\t{}\t{}\t{}\n'.format(cluster, answer_ids[i], i, files[i]))

                _major, _total = calc_purity(result_file, purity_file)
                major += _major
                total += _total

            # print (float(major)/total)
            purity.append(float(major)/total)

    purity = np.array(purity).reshape(loop, len(predictions))

    print(np.average(purity, axis=0), np.var(purity, axis=0))

    return np.average(purity, axis=0), np.var(purity, axis=0)



if __name__ == '__main__':
    avg, var = clustering_DsetMix()
    np.savetxt('clustering_DsetMix.csv', np.array(avg), delimiter=',')

    # avg, var = clustering_Sasaki(0)
    # np.savetxt('clustering_Sasaki_attn.csv', np.array(avg), delimiter=',')
    # clustering_Khuong()

    # avgs = []
    # vars = []
    # for i in range(5):
    #     avg, var = clustering_Khuong(i)
    #     avgs.append(avg)
    #     vars.append(var)
    #
    # np.savetxt('cluster_khuong_euclid.csv', np.array(avgs), delimiter=',')

    # avgs = []
    # vars = []
    # for i in range(5):
    #     avg, var = clustering_Sasaki(i)
    #     avgs.append(avg)
    #     vars.append(var)
    #
    # np.savetxt('cluster_sasaki_euclid.csv', np.array(avgs), delimiter=',')

