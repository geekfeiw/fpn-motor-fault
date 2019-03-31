import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import hdf5storage

from tqdm import tqdm
# from multi_scale_nores import *
# from multi_scale_one3x3 import *
# from multi_scale_one5x5 import *
# from multi_scale_one7x7 import *

batch_size = 512

#1
# mode = 'change'
# network = 'diapyra'
# pkl_name = 'ChaningSpeed_Train97.949Test94.485'
# caspyra = torch.load('weights/changingResnet/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
#2
# mode = 'change'
# network = 'nodiapyra'
# pkl_name = 'ChaningSpeed_Train95.480Test92.493'
# caspyra = torch.load('weights/changingResnet/'+ network + '/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
# #3
# mode = 'change'
# network = 'dianopyra'
# pkl_name = 'ChaningSpeed_Train93.360Test90.917'
# caspyra = torch.load('weights/changingResnet/'+ network + '/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
# #4
# mode = 'change'
# network = 'nodianopyra'
# pkl_name = 'ChaningSpeed_Train89.397Test86.468'
# caspyra = torch.load('weights/changingResnet/'+ network + '/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
#
#
# mode = 'still'
# network = 'nodianopyra'
# pkl_name = 'StillSpeed_Train95.475Test94.904'
# caspyra = torch.load('weights/NochangingResnet/'+ network + '/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
# #6
# mode = 'still'
# network = 'dianopyra'
# pkl_name = 'StillSpeed_Train97.730Test96.753'
# caspyra = torch.load('weights/NochangingResnet/'+ network + '/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
# #7
# mode = 'still'
# network = 'nodiapyra'
# pkl_name = 'StillSpeed_Train99.047Test98.560'
# caspyra = torch.load('weights/NochangingResnet/'+ network + '/'+ pkl_name + '.pkl')
# caspyra.cuda().eval()
# #8
mode = 'still'
network = 'diapyra'
pkl_name = 'StillSpeed_Train100.0Test99.618'
caspyra = torch.load('weights/NochangingResnet/'+ pkl_name + '.pkl')
caspyra.cuda().eval()

data = sio.loadmat('data/lrn/yunsumotor/stillSpeed_test.mat')
test_data = data['test_data_split']
test_label = data['test_label_split']

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
test_data = test_data.view(num_test_instances, 1, -1)
test_label = test_label.view(num_test_instances, 1)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

prediction_label = []
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels = labels.squeeze()
        labelsV = Variable(labels.cuda())
        # labelsV = labelsV.view(-1)

        predict_label_1, predict_label_2, predict_label_3, predict_label_4 = caspyra(samplesV)
        prediction = predict_label_1.data.max(1)[1]

        prediction_label.append(prediction.cpu().numpy())

hdf5storage.savemat('matfiles/' + mode + '_' + network + '_' + pkl_name+ '_prediction.mat', {'prediction_label': prediction_label})

