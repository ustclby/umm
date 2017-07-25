import glob
import os
import numpy as np
import h5py

def check_hist(folder):
    """ check the labels distribution of the training and testing data
        draw the histogram
        0 means not available
    """
    WD = folder
    file_names = glob.glob(os.path.join(WD, '*.h5'))
    new_label = []

    for fn in file_names:
        try:
            with h5py.File(fn, 'r') as h5_file:
                labels = h5_file['label'][:, :]
                if len(new_label) == 0 :
                    new_label = labels
                else:
                    new_label = np.concatenate((new_label, labels),axis = 0)
        except:
            print(fn)

    attr = new_label.shape[1]
    for i in range(attr):
        print(np.min(new_label[:,i]),np.max(new_label[:,i]))
        train_hist = np.histogram(new_label[:,i], bins=range(130))
        train_hist = list(train_hist)
        train_hist1 = list(train_hist[0])
        with open('hist.txt', 'a') as in_files:
            in_files.write(str(train_hist1))
            in_files.write('\n')



def main():
    with open('hist.txt', 'w') as in_files:
        pass
    # train_folder = '/home/umm/Train_h5/'
    # check_hist(train_folder)

    with open('hist.txt', 'a') as in_files:
        in_files.write('\n\n\n')

    test_folder = '/home/umm/Test_h5/'
    check_hist(test_folder)


if __name__ == "__main__":
    main()
