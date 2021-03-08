import json
import wikipedia
import annotate
import os
import shutil
from termcolor import colored
import math
import nltk
import re
import multiprocessing

def get_titles(file, all_titles):
    with open(file) as fd:
        loaded_json = json.load(fd)
    for titles in loaded_json.values():
        all_titles.extend(titles)
    #titles = ([data['title'] for data in loaded_json['data']])


def processing_wikihop_articles(sentences, title, path):
        max_ss_one_doc = annotate.MAX_SENTENCES_IN_ONE_DOCUMENT
        if len(sentences) > max_ss_one_doc:
            parts_number = math.ceil(len(sentences) / max_ss_one_doc)

            for k in range(int(parts_number)):
                if k<10:
                    part = "_part 00" + str(k)
                else:
                    part = "_part 0" + str(k)
                ner_dict = {}
                part_sentences = sentences[(max_ss_one_doc*k):(max_ss_one_doc*(k+1))]
                ner_dict = annotate.extract_ner(part_sentences, ner_dict, title+part, path)
                annotate.output(part_sentences, ner_dict, title, path, part)
        else:
            ner_dict = {}
            ner_dict = annotate.extract_ner(sentences, ner_dict, title+"_part 000", path)
            annotate.output(sentences, ner_dict, title, path, "_part 000")


def get_sentences(title):
    page = wikipedia.page(title)
    contents = nltk.tokenize.sent_tokenize(page.content.replace("\n", ' ').replace('\xa0', ' '))
    sentences = []
    for sentence in contents:
        if any(sb in sentence for sb in ("== External links ==", "== References ==", "== See also ==")):
            continue
        sentences.append(re.sub(r"==.*== ", "", sentence))
    return sentences


path = "wikihop"
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)

all_titles = []
print(colored("processing wikihop data...", 'magenta'))
get_titles("dev_titles.json", all_titles)
get_titles("train_titles.json", all_titles)

print(len(all_titles))
documents = list(set(all_titles))
print("retrieved "+str(len(documents))+" wikipedia documents.")


process_plan = list(annotate.chunks(range(0, len(documents)), annotate.ALLOWED_PARALLEL_PROCESS))

for i in range(len(process_plan)):
    process_pool = []
    for j in process_plan[i]:
        print(colored(("paragraph " + str(j) + " " + documents[j]), 'yellow'))
        sentences = get_sentences(documents[j])
        #processing_wikihop_articles(sentences, documents[j], path)
        p = multiprocessing.Process(target=processing_wikihop_articles, args=(sentences, documents[j], path, ))
        process_pool.append(p)
        p.start()
    for p in process_pool:
        p.join()
print("the end!")

