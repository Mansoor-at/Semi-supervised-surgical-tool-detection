
import os.path
import shutil
import os
import numpy as np


def get_files_from_folder(path):

    files = os.listdir(path)
    return np.asarray(files)
path_to_original = 'ut/val_xml'


path_to_save = 'ut/test_xml'
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
files = get_files_from_folder(path_to_original)
l=os.listdir('ut/test2017')
li=[x.split('.')[0] for x in l]
b=[x.split('.')[0] for x in files]
d=list(set(b) & set(li))
for i in range(len(d)):
    g=d[i]
    index= b.index(g)
    dst = os.path.join(path_to_save, files[index])
    src = os.path.join(path_to_original, files[index])
    shutil.move(src, dst)


'''
   # moves data
for j in files:
    dst = os.path.join(path_to_save, files[j])
    src = os.path.join(path_to_original, files[j])
    shutil.move(src, dst)
'''




