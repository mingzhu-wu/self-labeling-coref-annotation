import sys
import json
import os
import shutil

if __name__ == '__main__':
    input_file = sys.argv[1]
    file_object = open(input_file)
    loaded_json = json.load(file_object)
    path = input_file.split(".")[0]
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    paragraphs = ([data['paragraphs'] for data in loaded_json['data']])
    titles = ([data['title'] for data in loaded_json['data']])

    for j in range(len(paragraphs)):
        all_context = []
        for paraph in paragraphs[j]:
            context = paraph['context']
            all_context.append(context)
        ss_in_context = " ".join(all_context).replace('\n', ' ').replace('\xa0', ' ')
        try:
            with open(path+'/'+titles[j]+".txt", 'w') as f:
                f.write(ss_in_context)
        except BaseException as e:
            print("Can not open file path "+path+'/'+titles[j]+".txt")
