from stanfordcorenlp import StanfordCoreNLP
from termcolor import colored
import json
import sys
import os
import string
import re
from gender import GenderRecoginition
import spacy
import copy
import shutil
import time
import math
import multiprocessing
import random

ALLOWED_PARALLEL_PROCESS = 8
MAX_NER_TO_REPLACE_IN_ONE_SENTENCE = 3
MAX_SENTENCES_IN_ONE_DOCUMENT = 100
UKP_SERVER = 'http://krusty.ukp.informatik.tu-darmstadt.de'
UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"
LOCALHOST = "http://localhost"


class StanfordNLP(object):
    """docstring for StanfordNLP"""

    def __init__(self, host=UKP_SERVER_NED, port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=3000)
        # if add gender here, the gender informtion will be included in the coref results. so it should be used
        # together with dcoref.
        # in fact, even not specify gender in the annotators, as long as dcoref is added, there will be gender
        # information
        self.pros = {
            'annotators': 'tokenize, ssplit, truecase, pos, lemma, ner, parse',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.pros))

    def dcoref(self, sentence):
        return self.nlp.coref(sentence)


def output(ss, file, path):
    # file example: "China_part000"
    file_path = path + '/' + file
    for sentence in ss:
        try:
            # this assumes that there will not be two identical sentences
            mycorenlp(sentence, file_path)
        except BaseException as e:
            print(colored(e, 'red'))
            print(colored(("found exception in sentence: " + sentence), 'red'))
        else:
            continue


def mycorenlp(sentence, file):
    sta_nlp = StanfordNLP()
    # file example: wb/New York City_part000
    # doc_id example: wb/New/00/York/00/City
    doc_id = file.split("_")[0].replace(" ", "/00/")
    part_number = file.split('_')[1][-1]
    tokens = sta_nlp.word_tokenize(sentence)
    ner_ids = ['-']*len(tokens)
    final_result = []
    # file_name example: "wb/China_part000_annotate"
    file_name = file + "_annotate"
    for token in tokens:
        line_element = [doc_id, part_number, '-', token]
        line_element.extend(['-']*8)
        final_result.append(line_element)

    for i in range(MAX_NER_TO_REPLACE_IN_ONE_SENTENCE):
        new_file_name = file_name + "_" + str(i+1) + ".txt"
        new_result = copy.deepcopy(final_result)
        doc_id_postfix = '/00/annotate'+str(i+1)
        new_final_result = []
        for line in new_result:
            line[0] = line[0]+doc_id_postfix
            new_final_result.append(line)

        write_final_result(new_file_name, new_final_result)


def write_final_result(file_name, result):
    fo = open(file_name, 'a+')
    fo.write("\n")

    for n in range(len(result)):
        result[n][2] = n

    for line_elem in result:
        line_str = ["{}".format(element) for element in line_elem]
        line_width = [20, 5, 5, 30, 5, 5, 5, 5, 5, 5, 5, 5, 10]
        line = "  ".join(line_str[i].rjust(line_width[i]) for i in range(0, len(line_str)))
        fo.write(line)
        fo.write("\n")
    fo.close()


def get_sentences(paragraph):
    ss = []
    # parse json style data, get the context field, split each context into sentences, and process each sentences
    for paraph in paragraph:
        context = paraph['context']
        ss_in_context = context.replace('\n', ' ').replace('\xa0', ' ').split(". ")
        for sentence in ss_in_context:
            # the last sentence in each context don't need additional dot
            # this also assumes that won't be two identical sentences
            if ss_in_context.index(sentence) < len(ss_in_context) - 1:
                ss.append(sentence + '.')
            else:
                ss.append(sentence)
    return ss


def paragraph_process(sentences, paraph_name):
    # split one paragraph into several parts
    if len(sentences) > MAX_SENTENCES_IN_ONE_DOCUMENT:
        parts_number = math.ceil(len(sentences) / MAX_SENTENCES_IN_ONE_DOCUMENT)
        for k in range(int(parts_number)):
            part = "_part00" + str(k)
            part_sentences = sentences[(MAX_SENTENCES_IN_ONE_DOCUMENT*k):(MAX_SENTENCES_IN_ONE_DOCUMENT*(k+1))]
            output(part_sentences, paraph_name + part, path)
    else:
        output(sentences, paraph_name+"_part000", path)


if __name__ == '__main__':
    if os.path.exists('ner_spacy.txt'):
        os.remove('ner_spacy.txt')
    if os.path.exists("conll_style.txt"):
        os.remove('conll_style.txt')
    if os.path.exists('ner_result.txt'):
        os.remove('ner_result.txt')
    if os.path.exists('ner_stanford.txt'):
        os.remove('ner_stanford.txt')
    if os.path.exists('wb'):
        shutil.rmtree('wb')
    # "wb" means weblog, should be one of the genre in the conll
    path = "wb"
    os.makedirs(path)

    input_file = sys.argv[1]
    file_object = open(input_file)
    loaded_json = json.load(file_object)

    # parse json style data, get the context field, split each context into sentences, and process each sentences
    paragraphs = ([data['paragraphs'] for data in loaded_json['data']])
    titles = ([data['title'] for data in loaded_json['data']])
    # every title as a single file
    #lock = multiprocessing.Lock()
    random.shuffle(paragraphs)

    for i in range(len(paragraphs[:50])):
        print(colored(("paragraph " + str(i) + " " + titles[i]), 'yellow'))
        sentences = get_sentences(paragraphs[i])
        paragraph_process(sentences, titles[i])






