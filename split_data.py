import random
import os
import sys

# spath: data, dpath: train or dev or test, filename: Catholic Church_part001
def move_file(spath, dpath, filename):
    with open(spath+'/'+dpath+'_annotate.txt', "a+") as ofile:
        if os.path.exists(spath+'/'+filename+'_annotate.txt'):
            with open(spath+'/'+filename+'_annotate.txt', 'r') as nfile:
                for line in nfile.readlines():
                    ofile.write(line)
            ofile.write("\n")


def split_data(data_path):
    files = os.listdir(data_path)
    paragraphs = []
    # get the title name of all the paragraphs
    for file in files:
        filename = file[:-4]
        if "annotate" not in filename:
            paragraphs.append(filename)
    #random.shuffle(paragraphs)
    i = 0
    # move the file to the new directory, title: New York City_part000
    print(len(paragraphs))
    for title in paragraphs:
        if i < int(len(paragraphs) * 0.7):
            move_file(data_path, "train", title)
        #elif i < int(len(paragraphs) * 0.9):
           # move_file(data_path, "dev", title)
        else:
            move_file(data_path, "dev", title)
        i += 1


def concatenate_annotations(data_path):
    files = os.listdir(data_path)
    paragraphs = []
    # get the title name of all the paragraphs
    for file in files:
        filename = file[:-4]
        if "annotate" not in filename:
            paragraphs.append(filename)
    i = 0
    # move the file to the new directory, title: New York City_part000
    print(len(paragraphs))
    for title in paragraphs:
        move_file(data_path, "train", title)

if __name__ == '__main__':
    path = sys.argv[1]
    split_data(path)

    #concatenate_annotations(path)
    # concatenate_annotations("wb/dev")
    # concatenate_annotations("wb/test")



