import numpy as np
from dicompylercore import dicomparser
import cv2

def convert_dcm_png(root, save_root):
    for folder in tqdm(os.listdir(root)):
        for subfolder in os.listdir(os.path.join(root, folder)):
            for file in os.listdir(os.path.join(root, folder, subfolder)):
                img_dcm = dicomparser.DicomParser(os.path.join(root, folder, subfolder, file))
                cv2.imwrite(os.path.join(save_root, file.split('dcm')[0]+'png'), np.array(img_dcm.GetImage()))

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle