import pandas as pd
import torch
import numpy as np
import torchaudio
import requests
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd  # To play sound in the notebook
import os


SAVEE = '/Users/devpatelio/Downloads/Coding/Python/pyTorch/audio_mood/ALL'
RAV =  '/Users/devpatelio/Downloads/Coding/Python/pyTorch/audio_mood/ravdess-emotional-speech-audio/audio_speech_actors_01-24'
TESS = '/Users/devpatelio/Downloads/Coding/Python/pyTorch/audio_mood/TESS Toronto emotional speech set data'
CREMA = '/Users/devpatelio/Downloads/Coding/Python/pyTorch/audio_mood/AudioWAV'


dirlist_SAVEE = os.listdir(SAVEE)
emotion_SAVEE = []
path_SAVEE = []

def PrintThree(filename):
    return filename[-8:-6]


for i in dirlist_SAVEE:
    if PrintThree(i)=='_a':
        emotion_SAVEE.append('male_angry')
    elif PrintThree(i)=='_d':
        emotion_SAVEE.append('male_disgust')
    elif PrintThree(i)=='_f':
        emotion_SAVEE.append('male_fear')
    elif PrintThree(i)=='_h':
        emotion_SAVEE.append('male_happy')
    elif PrintThree(i)=='sa':
        emotion_SAVEE.append('male_sad')
    elif PrintThree(i)=='su':
        emotion_SAVEE.append('male_surprise')
    elif PrintThree(i)=='_n':
        emotion_SAVEE.append('male_neutral') 
    path_SAVEE.append(SAVEE + '/' + i)

SAVEE_df = pd.DataFrame(emotion_SAVEE, columns=['labels'])
SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path_SAVEE, columns=['path'])], axis=1)
SAVEE_df.labels.value_counts()




dirlist_RAV = os.listdir(RAV)
actor_range = list(range(1, 25))
# for i in actor_range:
#     actor_range[i] = str(i+1)

# for i in actor_range[0:9]:
#     actor_range[i] = str(i+1).zfill(2)
# actor_range[0] = str(actor_range[0]).zfill(2)
# str_actor_range = [str(item) for item in actor_range]
# print(str_actor_range)

gender_RAV = []
emotion_RAV = []
path_RAV = []

for subdir in dirlist_RAV:
    for file in os.listdir(RAV + '/' + subdir):
        if str(file[6:8]) == '01':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_neutral')
        elif str(file[6:8]) == '02':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_calm')
        elif str(file[6:8]) == '03':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_happy')
        elif str(file[6:8]) == '04':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_sad')
        elif str(file[6:8]) == '05':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_angry')
        elif str(file[6:8]) == '06':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_fearful')
        elif str(file[6:8]) == '07':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_disgust')
        elif str(file[6:8]) == '08':
            if int(file[18:20])%2==0:
                temp = 'female'
            else: 
                temp = 'male'
            gender_RAV.append(temp)
            emotion_RAV.append(temp+'_surprised')
        path_RAV.append(RAV + '/' + subdir + '/' + file)
        

RAV_df = pd.DataFrame(emotion_RAV, columns=['labels'])
RAV_df['source'] = 'RAVDESS'
RAV_df = pd.concat([RAV_df, pd.DataFrame(path_RAV, columns=['path'])], axis=1)
RAV_df.labels.value_counts()




dirlist_TESS = os.listdir(TESS)
dirlist_TESS.sort()
dirlist_TESS

path_TESS = []
emotion_TESS = []

for i in dirlist_TESS:
    for file in os.listdir(TESS + '/' + i):
        if i == 'OAF_angry' or i=='YAF_angry':
            emotion_TESS.append('female_angry')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotion_TESS.append('female_disgust')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotion_TESS.append('female_fear')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotion_TESS.append('female_happy')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotion_TESS.append('female_neutral')                                
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotion_TESS.append('female_surprise')               
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotion_TESS.append('female_sad')
        else:
            emotion_TESS.append('Unknown')
        path_TESS.append(TESS + '/' + i + '/' + file)

TESS_df = pd.DataFrame(emotion_TESS, columns=['labels'])
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df, pd.DataFrame(path_TESS, columns=['path'])], axis=1)
TESS_df.labels.value_counts()




dirlist_CREMA = os.listdir(CREMA)
dirlist_CREMA.sort()

gender_CREMA = []
emotion_CREMA = []
path_CREMA = []
female_CREMA = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

for i in dirlist_CREMA:
    if int(i[0:4]) in female_CREMA:
        temp = 'female'
    else: 
        temp = 'male'
    gender_CREMA.append(temp)
    
    if i[9:12] == 'SAD' and temp == 'male':
        emotion_CREMA.append('male_sad')
    elif i[9:12] == 'SAD' and temp == 'female':
        emotion_CREMA.append('female_sad')
    elif i[9:12] == 'ANG' and temp == 'female':
        emotion_CREMA.append('female_angry')
    elif i[9:12] == 'ANG' and temp == 'male':
        emotion_CREMA.append('male_angry')
    elif i[9:12] == 'DIS' and temp == 'male':
        emotion_CREMA.append('male_disgust')
    elif i[9:12] == 'DIS' and temp == 'female':
        emotion_CREMA.append('female_disgust')
    elif i[9:12] == 'FEA' and temp == 'female':
        emotion_CREMA.append('female_fear')
    elif i[9:12] == 'FEA' and temp == 'male':
        emotion_CREMA.append('male_fear')
    elif i[9:12] == 'HAP' and temp == 'female':
        emotion_CREMA.append('female_happy')
    elif i[9:12] == 'HAP' and temp == 'male':
        emotion_CREMA.append('male_happy')
    elif i[9:12] == 'NEU' and temp == 'female':
        emotion_CREMA.append('female_neutral')
    elif i[9:12] == 'NEU' and temp == 'male':
        emotion_CREMA.append('male_neutral')
    path_CREMA.append(CREMA + '/' + i)
    
CREMA_df = pd.DataFrame(emotion_CREMA, columns=['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df, pd.DataFrame(path_CREMA, columns=['path'])], axis=1)
CREMA_df.labels.value_counts()




dataset = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis=0)
print(dataset.labels.value_counts())
dataset.to_csv("audio_stuff.csv", index=False)

dataset.iloc[1]




filename = dataset.iloc[796]['path']
wf, sr = librosa.load(filename, res_type='kaiser_fast')
mfcc = librosa.feature.mfcc(y=wf, sr=sr)

plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(wf, sr=sr)

plt.figure(figsize=(20, 15))
plt.subplot(3, 1, 1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.xlabel('time')
plt.colorbar()
print(dataset.iloc[796]['labels'])


ipd.Audio(filename)
print(wf.shape)
print(mfcc.shape)



filename = dataset.iloc[486]['path']
wf, sr = librosa.load(filename, res_type='kaiser_fast')
mfcc = librosa.feature.mfcc(y=wf, sr=sr)

plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(wf, sr=sr)

plt.figure(figsize=(20, 15))
plt.subplot(3, 1, 1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.xlabel('time')
plt.colorbar()
print(dataset.iloc[486]['labels'])

print(wf.shape)
print(mfcc.shape)




mfcc_list = list()

for filename in dataset['path']:
    wf, sr = librosa.load(filename, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=wf, sr=sr)
    mfcc_list.append(mfcc)
#     for i in mfcc:
#         empty_mean_mfcc.append(np.array(np.mean(i)))
        




print(f"MFCC_LIST FIRST ELEMENT: {mfcc_list[0][0]}")




mean_list = list()
for i in mfcc_list:
    mean_item = list()
    for x in i: 
        mean_item.append(np.mean(x))
    mean_list.append(mean_item)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim as optim, functional as F
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

""
for counter, i in enumerate(mean_list):
    for count, x in enumerate(i):
        mean = np.mean(i, axis=0)
        std = np.std(i, axis=0)
        x = (x - mean)/std
        mean_list[counter][count] = x
    




dataset['mfcc'] = mean_list
dataset.head()




dataset.to_csv('processed_dataset.csv', index=0)




labels = []




from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim as optim, functional as F
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

label_column = dataset['labels']
labels = (label_column.unique())
one_hot_template = torch.zeros(len(labels), len(labels))

for i in range(0, len(labels)):
    one_hot_template[i][i] = 1

label_encoder_dict = {labels[i]: one_hot_template[i] for i in range(0, len(labels))}
dataset_np = np.array(dataset['mfcc'])
labels_np = np.array(dataset['labels'])


        

combined = []
for counter, x in enumerate(labels_np):
    x = label_encoder_dict[str(x)]
    combined.append(  [dataset_np[counter], x]  )





for counter, i in enumerate(combined):
    combined[counter][0] = torch.FloatTensor(combined[counter][0])




import matplotlib.pyplot as plt

categories = labels
count = []

for i in categories:
    count.append(dataset.loc[dataset['labels'] == i, 'labels'].count())




plt.figure(figsize=(30, 10))
plt.bar(categories, count)




np.random.shuffle(combined)
train_index = int(len(combined)*0.8)
train_set_np = combined[:train_index]
test_set_np = combined[train_index:-1]

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import torchvision.transforms as transforms

trainloader = torch.utils.data.DataLoader(train_set_np, shuffle=False, batch_size = 60)
testloader = torch.utils.data.DataLoader(test_set_np, shuffle=False, batch_size = 60)

train = iter(trainloader)
mfcc_data, label = next(train)




print(mfcc_data.view(2, -1).shape)