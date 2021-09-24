# Copyright 2021 Fedlearn authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import time
import torch
from model_sphereface import sphere20a
THRESHOLD = 0.295

def get_cosdist(f1: np.ndarray, f2: np.ndarray):
    if isinstance(f1, list):
        f1 = np.asarray(f1)
    if isinstance(f2, list):
        f2 = np.asarray(f2)
    # print(f1.shape, f2.shape)
    return f1.dot(f2) / ( np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)



class Insecure_Client(object):

    def __init__(self):
        self.torch_model = sphere20a(feature=True).cpu()
        pretrained_weights = torch.load('../../data/FaceRecognition/sphere20a_20171020.pth')
        pretrained_weights_for_inference = {k:v for k, v in pretrained_weights.items() if 'fc6' not in k}
        self.torch_model.load_state_dict(pretrained_weights_for_inference )

    def inference(self, raw_img):
        t0 = time.time()
        x = torch.tensor(raw_img).cpu()
        _prob = self.torch_model(x).detach().numpy()
        cosdist = get_cosdist(_prob[0], _prob[1])
        return {'feature': _prob, 'dist': cosdist, 'pred': int(cosdist > THRESHOLD), 'runtime': time.time()-t0}



def get_input(n=1000):

    with open('../../data/FaceRecognition/LFW/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    img_label = []
    for i in range(n):
        p = pairs_lines[i].replace('\n','').split('\t')

        if 3==len(p):
            sameflag = 1
            name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
            name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
        if 4==len(p):
            sameflag = 0
            name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
            name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

        img1 = cv2.imread("../../data/FaceRecognition/LFW/lfw_processed/"+name1)
        img2 = cv2.imread("../../data/FaceRecognition/LFW/lfw_processed/"+name2)
        img1_normalized = (img1.transpose(2, 0, 1)-127.5)/128.0
        img2_normalized = (img2.transpose(2, 0, 1)-127.5)/128.0

        img_label.append( [np.stack([img1_normalized, img2_normalized], 0).astype('float32'), sameflag] )
    return img_label



if __name__ == '__main__':
    insecure_client = Insecure_Client()
    raw_img_set = get_input(20)  #
    correct = 0
    for i, (raw_img, sameflag) in enumerate(raw_img_set):
        ref = insecure_client.inference(raw_img)
        print("label: %r;   Pred: %r;   Time: %.2fs;   Dist: %.12f" % ( sameflag,  ref['pred'], ref['runtime'], ref['dist']) )
