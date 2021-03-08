import sys
import json
import os
import shutil

if __name__ == '__main__':
    input_file = sys.argv[1]
    file_object = open(input_file)
    loaded_json = json.load(file_object)
    path = input_file[:-5]
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    supports_list = ([data_dict["supports"] for data_dict in loaded_json])
    ids = ([data_dict["id"] for data_dict in loaded_json])
    doc_number = len(supports_list)

    for j in range(doc_number):
        passages = []
        sentences = []
        for passage in supports_list[j]:
            passages.append(passage)
        try:
            with open(path+'/'+ids[j]+".txt", 'w') as f:
                f.write(" ".join(passages))
        except BaseException as e:
            raise e
            print("Can not open file path "+path+'/'+ids[j]+".txt")

