
import cv2
import numpy as np
import os

def mergeList(input_list):
    merge = []
    for item in input_list:
        merge += item
    return merge

def get_file_tag(file_path, data_path):
    return file_path.replace(data_path + '\\', '').split('.')[0]

def drawImage(fileName, traces, out_height=128):
    print(fileName)
    if traces == None: return

    padding = 5

    out_fg = out_height - 2 * padding + 1

    serialize = mergeList(traces)
    serialize = np.array(serialize)

    # print(serialize.shape)

    coord_size = np.max(serialize, axis=0) - np.min(serialize, axis=0)

    off_set = np.min(serialize, axis=0)

    # print(coord_size)

    ratio = float(out_fg) / (coord_size[1] + 1e-10)

    img_size = (coord_size * ratio).astype(int)
    # print(img_size)

    img = np.zeros((img_size[1] + 2 * padding, img_size[0] + 2 * padding, 3), np.uint8)

    for trace in traces:

        if len(trace) == 0: continue

        trace = (np.array(trace) - off_set) * ratio + padding
        trace = trace.astype(int)

        trace_shifted = np.concatenate((trace[0:1], trace[:-1]), axis=0)
        for pt1, pt2 in zip(trace_shifted, trace):
            # print(pt1, pt2)
            cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), color=(255,255,255), thickness=2)


    # cv2.imshow("foo",img)
    os.makedirs(os.path.dirname(fileName), exist_ok=True)
    cv2.imwrite(fileName, img)
    # cv2.waitKey()

def read_strokes_json(inkFile):
    import json
    out = json.load(open(inkFile, 'r'))
    print (inkFile, out)

    strokes = []
    for key, value in out.items():
        if key == 'stroke':
            strokeGroup = value
            for stroke in strokeGroup:
                print (stroke)
                strokes.append (list(zip(stroke['x'], stroke['y'])))


    # print(strokes)
    # print (out)

    # fileName = inkFile.split('\\')[-1]
    # drawImage('images' + '\\' + fileName + '.png', strokes)
    return strokes



def convertFile(zip_input):
    inkFile, img_path, data_path = zip_input
    traces = read_strokes_json(inkFile)
    drawImage(img_path + '\\' + get_file_tag(inkFile, data_path) + '.png', traces)

def convertJson():
    data_path = 'X:\\HandwrittenDatabase_Yasuno'
    img_path = 'images'

    import glob
    files = glob.glob(data_path + '\\**\\*.json', recursive=True)[:]
    print (len(files))

    from multiprocessing import Pool
    with Pool(20) as p:
        p.map(convertFile, zip(files, [img_path]*len(files), [data_path] * len(files)))

# def test():


if __name__ == '__main__':
    # drawImage()
    # convertInkml()
    convertJson()

    # read_strokes_json('Z:\\hand_mid_div-3.wdgt\\logs\\2019_06_27_04_20_57')
