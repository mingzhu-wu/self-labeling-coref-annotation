from termcolor import colored
import json
import sys
import os
import shutil
import multiprocessing
import annotate
import nltk
from mycorenlp import MyCoreNLP
import math


def get_sentences(support_list):
    sts = []
    for passage in support_list:
        sts.extend(nltk.tokenize.sent_tokenize(passage))
    return sts


def get_all_sentences(input_file):
    input_file = sys.argv[1]
    file_object = open(input_file)
    loaded_json = json.load(file_object)
    # parse json style data, get the context field, split each context into sentences, and process each sentences
    supports = ([data_dict["supports"] for data_dict in loaded_json])

    ss_in_file = []
    with open("sentences_in_wikihop_dev.txt", "a+") as f:
        # unique_ss = f.readlines()
        for support in supports:
            for s in get_sentences(support):
                if s not in ss_in_file:
                    f.write(s)
                    f.write("\n")
                    ss_in_file.append(s)
                else:
                    continue


def process(sentence, doc_id, part, fp):
    sta_nlp = annotate.MyCoreNLP()
    # part: _part 002 or _part 011
    part_number = int(part[7:])
    tokens = sta_nlp.word_tokenize(sentence)
    poss = sta_nlp.pos_tag(sentence)
    assert(len(tokens) == len(poss))

    final_result = []
    for i in range(len(tokens)):
        token = tokens[i]
        pos = poss[i][1]
        line_element = [doc_id, part_number, i, token, pos, '*', '-', '-', '-', '-', '*', '-']
        final_result.append(line_element)

    fp.write("\n")
    for line_elem in final_result:
        line_str = ["{}".format(element) for element in line_elem]
        line_width = [20, 5, 5, 30, 5, 5, 5, 5, 5, 5, 5, 5, 10]
        line = "  ".join(line_str[i].rjust(line_width[i]) for i in range(0, len(line_str)))
        fp.write(line)
        fp.write("\n")


def output(ss, doc_id, path, part):
    # file example: "New York City"
    file_path = path + '/' + doc_id
    fo = open(file_path, 'w')
    fo.write("#begin document ("+doc_id+"); "+part[1:])
    for sentence in ss:
        try:
            # this assumes that there will not be two identical sentences
            if sentence == " ":
                continue
            process(sentence, doc_id, part, fo)
        except BaseException as e:
            print(colored(e, 'green'))
            print(colored(("found exception in sentence: " + sentence, "should not happen very offen"), 'green'))
            #raise e
        else:
            continue
    fo.write("\n#end document")
    fo.close()


def paragraph_process(sentences, paraph_name, path, doc):
    ner_dict = {}
    ner_dict = annotate.extract_ner(sentences, ner_dict, paraph_name+"_part 000", path)
    doc_id = paraph_name + ".txt"
    file_path = path + '/' + doc_id
    annotate.output(sentences, ner_dict, doc_id, "_part 000", file_path, doc)


if __name__ == '__main__':
    if os.path.exists('wikihop'):
        shutil.rmtree('wikihop')

    path = "wikihop"
    os.makedirs(path)
    input_file = sys.argv[1]
    #get_all_sentences(input_file)

    file_object = open(input_file)
    loaded_json = json.load(file_object)
    # parse json style data, get the context field, split each context into sentences, and process each sentences
    supports_list = ([data_dict["supports"] for data_dict in loaded_json])
    ids = ([data_dict["id"] for data_dict in loaded_json])
    doc_number = len(supports_list)
    print(doc_number)
    nlp = MyCoreNLP()

    #process_plan = list(annotate.chunks(range(0, doc_number), 16))
    pool = multiprocessing.Pool(processes=annotate.ALLOWED_PARALLEL_PROCESS)
    for j in range(doc_number):
        #for j in process_plan[i]:
        print(colored(input_file + " document " + str(j), 'yellow'))
        passages = []
        sentences = []
        for passage in supports_list[j]:
            passages.append(passage)
        sentences = nlp.ssplit(" ".join(passages))
        # generate conll style file without any annotations
        # doc_id = ids[j] + ".txt"
        # output(sentences, ids[j]+".txt", path, "_part 000")
        # paragraph_process(sentences, ids[j], path, " ".join(passages))
        #pool.apply_async(output, (sentences, ids[j]+".txt", path, "_part 000 ", ))
        pool.apply_async(paragraph_process, (sentences, ids[j], path, " ".join(passages), ))
    pool.close()
    pool.join()
    print("the end!")






