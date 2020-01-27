
from xml.dom.minidom import parse

'''
<traceGroup xml:id="3">
	<annotation type="truth">From ITF</annotation>
	<traceGroup xml:id="4">
		<annotation type="truth">\{</annotation>
		<traceView traceDataRef="0"/>
		<annotationXML href="{_1"/>
	</traceGroup>
	<traceGroup xml:id="5">
		<annotation type="truth">u</annotation>
		<traceView traceDataRef="1"/>
		<annotationXML href="u_1"/>
	</traceGroup>
	<traceGroup xml:id="6">
		<annotation type="truth">\}</annotation>
		<traceView traceDataRef="2"/>
		<annotationXML href="}_1"/>
	</traceGroup>
</traceGroup>
</ink>
'''


def get_syms_seq(inkmlfile):
    rootInk = parse(inkmlfile)
    # first trace group
    annotation = rootInk.getElementsByTagName('traceGroup')

    syms_list = []

    for sym in annotation[0].getElementsByTagName('traceGroup'):
        sym_char = sym.getElementsByTagName('annotation')[0].firstChild.data
        sym_order = [int(ref.getAttribute('traceDataRef')) for ref in sym.getElementsByTagName('traceView')]
        syms_list.append((sym_char, sym_order))

    return syms_list

def get_latex_truth(inkmlfile):
    rootInk = parse(inkmlfile)
    # first trace group
    annotations = rootInk.getElementsByTagName('annotation')

    for annotation in annotations:
        if annotation.getAttribute('type') == 'truth':
            return annotation.firstChild.data

    return ''

def get_file_tag(file_path, data_path):
    return file_path.replace(data_path + '\\', '')

def create_supervised_label(raw_data):
    sup_data = []

    for (file_tag, syms_seq) in raw_data:
        # print (file_tag, syms_seq)
        skip = False
        for (sym, sym_order) in syms_seq:
            # assert sym_order != [], 'Annotation error empty id ' + file_tag
            if sym_order == []:
                print ('Skip {}: empty id '.format(file_tag))
                skip = True
        if skip:
            continue
        syms_seq.sort(key=lambda r: min(r[1]))
        # print (list(zip(*syms_seq))[0])
        sup_data.append((file_tag, list(zip(*syms_seq))[0]))
    return sup_data

def write_supervised_data(sup_data, delimiter=''):
    line_data = []

    for (file_tag, sym_seq) in sup_data:
        line_data.append(file_tag.split('.')[0] + '\t' + delimiter.join(sym_seq) + '\n')
    return  line_data

def _create_annotation_full():
    data_path = 'D:\\database\\CROHME\\CROHME_2016_for_clustering\\inkml\\valid'

    import glob
    files = glob.glob(data_path + '\\*.inkml')[:]
    print (len(files))

    out = [(get_file_tag(file, data_path), get_syms_seq(file)) for file in files]

    import json
    with open('crohme2016_valid_annotation_full.json', 'w') as f:
        json.dump(out, f)


def _create_annotation_sym_seq():
    import json
    out = json.load(open('crohme2016_valid_annotation_full.json', 'r'))

    sup_data = create_supervised_label(out)

    with open('crohme2016_valid_annotation_sym_seq.txt', 'w') as f:
        f.writelines(write_supervised_data(sup_data))

def _create_annotation_latex():
    data_path = 'D:\\database\\CROHME\\CROHME_2016\\test\\TEST2016_INKML_GT'

    import glob
    files = glob.glob(data_path + '\\*.inkml')[:]
    print (len(files))

    out = [(get_file_tag(file, data_path), get_latex_truth(file)) for file in files]
    with open('crohme2016_test_annotation_latex.txt', 'w') as f:
        f.writelines(write_supervised_data(out))

def get_cluster_label_Khuong(file_name):
    return file_name.split('\\')[-1].split('.')[0]

def get_cluster_label_Sasaki(file_name):
    return file_name.split('\\')[-1].split('_')[0]

def _create_annotation():
    data_path = 'D:\\Workspace\\math_clustering_wcnn\\data\\Khuong\\imgs_h128'
    # data_path = 'D:\\Workspace\\math_clustering_wcnn\\data\\Sasaki\\imgs'
    # data_path = ''

    import glob
    files = glob.glob(data_path + '\\**\\*.png', recursive=True)[:]
    print (len(files))

    out = [(get_file_tag(file, data_path), get_cluster_label_Khuong(file)) for file in files]
    with open('khuong_test_annotation_group.txt', 'w') as f:
        f.writelines([data[0].split('.')[0] + '\t' + data[1] + '\n' for data in out])


def check_latex():
    data_list = 'FileName_GroundTruth_List.txt'
    lines = open(data_list).readlines()


    latex = [map(lambda  s: s.replace('\\\\', '\\').replace('<', '\\lt').replace('>', '\\gt'), line.strip().split('\t')[1].split()) for line in lines]

    from itertools import chain


    latex_syms = list(set(chain.from_iterable(latex)))
    print (latex_syms)

    vocab = [line.strip() for line in open('vocab_syms.txt').readlines()]

    print ([i for i in latex_syms if i not in vocab])

def _create_annotation_syms():
    annotation_group = 'khuong_test_annotation_group.txt'
    groundtruth_latex = 'data/Khuong/groundtruth_latex.txt'


    # annotation_group = 'sasaki_test_annotation_group.txt'
    # groundtruth_latex = 'data/Sasaki/groundtruth_latex.txt'

    files_groups = [line.strip().split('\t') for line in open(annotation_group).readlines()]

    groups_latexs = [(line.strip().split('\t')[0], (line.strip().split('\t')[1])) for line in open(groundtruth_latex).readlines()]

    groups, latexs = list(zip(*groups_latexs))

    # check latex
    # from itertools import chain
    #
    # latex_syms = list(set(chain.from_iterable(latexs)))
    # print (latex_syms)
    #
    # vocab = [line.strip() for line in open('vocab_syms.txt').readlines()]
    #
    # print ([i for i in latex_syms if i not in vocab])
    #
    # return

    def convert(str):
        str =  str.replace('<', '\\lt').replace('>', '\\gt').replace('\\Re', 'R').replace('\\lor', 'v').replace('\\frac', '-').replace('{', '').replace('}', '').replace('^', '')
        str = ' '.join(list(set(str.split())))
        return str
    sym_seqs = [convert(latex) for latex in latexs]

    with open('khuong_annotation_sym_seq.txt', 'w') as f:
        f.writelines([file + '\t' + sym_seqs[groups.index(group)] + '\n' for file, group in files_groups])


def get_writer_Khuong(file_name):
    return int(file_name.split('\\')[-2].split('-')[0])

def get_writer_Sasaki(file_name):
    return int(file_name.split('\\')[-1].split('.')[0].split('_')[1])

def _shuffle_writers():
    files_syms = [line.strip().split('\t') for line in open('sasaki_annotation_sym_seq.txt').readlines()]

    files, syms = list(zip(*files_syms))

    writers = list(map(get_writer_Sasaki, files))
    writer_set = list(set(writers))

    import random
    random.shuffle(writer_set)
    print (writer_set)

    writer_to_cross_idx = {writer: id % 5 for id, writer in enumerate(writer_set)}

    cross_ids = [writer_to_cross_idx[writer] for writer in writers]
    print (cross_ids)

    with open('sasaki_annotation_sym_seq_cross.txt', 'w') as f:
        f.writelines([file + '\t' + sym + '\t' + str(cross_id) + '\n' for file, sym, cross_id in zip(files, syms, cross_ids)])


def main():
    # _create_annotation_full()
    # _create_annotation_sym_seq()
    # _create_annotation_syms()
    # _shuffle_writers()

    # print (get_latex_truth("D:\\database\\CROHME\\CROHME_2016_for_clustering\\inkml\\valid\\18_em_2.inkml"))
    _create_annotation_latex()
    exit(0)

main()