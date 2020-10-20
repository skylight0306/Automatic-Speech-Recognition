import os
import random 
import numpy as np
###################################

def copyFile(fileDir,tarDir):
    #取得所有圖片路徑
    pathDir = os.listdir(fileDir)
    file0 = np.load( fileDir + pathDir[0] )#bed
    file1 = np.load( fileDir + pathDir[1] )#cat
    file2 = np.load( fileDir + pathDir[2] )#happy

    arr = []
    trainYcat = []
    trainY = []
    #len(arr)
    length = np.zeros(3)
    total = len(file0) + len(file1)+ len(file2)
    for i in range(total): 
        num = random.randint(0,2)
        if num == 0 and length[0] < len(file0): 
            arr.append(file0[int(length[0])]) 
            length[0] = length[0] + 1
            trainYcat.append([1,0,0])
            trainY.append([0])
            
        elif num == 1 and length[1] < len(file1):
            arr.append(file1[int(length[1])])
            length[1] = length[1] + 1
            trainYcat.append([0,1,0])
            trainY.append([1])
            
        elif num == 2 and length[2] < len(file2):
            arr.append(file2[int(length[2])])
            length[2] = length[2] + 1
            trainYcat.append([0,0,1])
            trainY.append([2])

    
    # 進行非重置抽樣
    np.save(tarDir + 'trainX.npy', arr)
    np.save(tarDir + 'trainYcat.npy', trainYcat)
    np.save(tarDir + 'trainY.npy', trainY)
    
#路徑最後記得加上反斜線
fileDir = 'D:/MFCC_CNN/data/'  # 來源路徑
tarDir = 'D:/MFCC_CNN/data/' # 目的路徑
#file_choice_number = 8 # 挑選張數，不可超過
copyFile(fileDir,tarDir)