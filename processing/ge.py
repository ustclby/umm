import glob
import os

WD = r'/home/umm/Train_h5/'
files = glob.glob(os.path.join(WD, '*'))
with open('train.txt', 'w') as in_files:
    in_files.writelines(fn + '\n' for fn in files)


WD2 = r'/home/umm/Test_h5/'
files = glob.glob(os.path.join(WD2, '*'))
with open('test.txt', 'w') as in_files:
    in_files.writelines(fn + '\n' for fn in files)
