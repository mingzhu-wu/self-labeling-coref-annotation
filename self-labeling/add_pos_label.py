from mycorenlp import MyCoreNLP
from termcolor import colored
import annotate
import os
import shutil


# self-labeling wikicoref data, extract all the documents and sentences in the given wikicoref file
# add self-labeled annotations to it and generate new documents.
def self_labeling_wikicoref_doc():
    path = "WikiCoref/Documents"
    files = os.listdir(path)
    j = 0
    nlp = MyCoreNLP()

    if os.path.exists('wb'):
        shutil.rmtree('wb')
        # "wb" means weblog, should be one of the genre in the conll
    os.makedirs("wb")

    for file in files:
        print(colored(("file " + str(j) + " " + file), 'yellow'))
        j += 1
        with open(path+"/"+file) as f:
            doc = f.read()
            #print(doc)
        sentences = nlp.ssplit(doc)
        print(len(sentences))
        annotate.paragraph_process(sentences, file, "wb")


def get_doc_wikicoref():
    documents = []
    titles = []
    tokens = []
    with open("key-OntoNotesScheme.parsed") as f:
        for line in f.readlines():
            if line.startswith("#begin"):
                titles.append(line.strip("\n")[16:])
                continue
            if line.startswith("#"):
                documents.append(" ".join(tokens))
                tokens = []
            elif line == '\n':
                tokens.append("\n")
            else:
                tokens.append(line.split()[3])
    return documents, titles


def self_labeling_wikicoref():
    #"wb" means weblog, should be one of the genre in the conll
    path = "wb"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    documents, titles = get_doc_wikicoref()
    print(len(documents), len(titles))
    i = 0
    nlp = MyCoreNLP()
    for doc in documents:
        ner_dict = {}
        sentences = nlp.ssplit(doc)
        print("number of sentences: ", len(sentences))
        print(colored(("paragraph " + str(i) + " " + titles[i]), 'yellow'))
        if titles[i] != "Barack Obama":
            i+=1
            continue

        doc_id = path + '/' + titles[i].replace(" ", "/00/")
        file_path = path + '/' + titles[i] + "_part 000_annotate.txt"
        annotate.extract_ner(sentences, ner_dict, titles[i]+"_part 000", path)
        annotate.output(sentences, ner_dict, doc_id, "_part 000", file_path, doc)
        i += 1


# add doc_id and more column to the wikicoref data
def add_more_column_to_wikicoref():
    with open("key-OntoNotesScheme.v4_gold_conll", 'a+') as fw:
        with open("key-OntoNotesScheme.parsed") as fr:
            for line in fr.readlines():
                # if ss_index < len(sentences):
                #     poss = sta_nlp.pos(sentences[ss_index])
                #begin document (wb/Chordate); part 000
                if line.startswith("#begin"):
                    doc_id = "wb/"+ line.strip("\n")[16:].replace(" ", "/00/")
                    fw.write("#begin document (" + doc_id + "); part 000")
                    fw.write("\n")
                elif line == '\n' or line.startswith("#end"):
                    fw.write(line)
                    #ss_index += 1
                    #word_index = 0
                else:
                    # print(poss)
                    # print("word_index:", word_index)
                    # print("sentence: ", sentences[ss_index])
                    fw.write(generate_new_line(line, doc_id))
                    fw.write("\n")


def generate_new_line(line, doc_id):
    new_line = line.split()[:-1]
    new_line.extend(['-', '-', '-', '-', '*'])
    new_line.append(line.split()[-1])
    new_line[0] = doc_id

    line_str = ["{}".format(element) for element in new_line]
    line_width = [20, 5, 5, 30, 5, 20, 5, 5, 5, 5, 5, 5]
    final_line = "  ".join(line_str[i].rjust(line_width[i]) for i in range(0, len(line_str)))
    return final_line


def replace_pos(line, pos):
    new_line = line.split()
    if line.split()[3] == pos[0]:
        new_line[4] = pos[1]
    else:
        print(colored(line.split()[3], 'red'))
        print(colored(pos[0], 'yellow'))
    line_str = ["{}".format(element) for element in new_line]
    line_width = [20, 5, 5, 30, 5, 5, 5, 5, 5, 5, 5, 5, 10]
    final_line = "  ".join(line_str[i].rjust(line_width[i]) for i in range(0, len(line_str)))
    return final_line


if __name__ == '__main__':
    self_labeling_wikicoref()
    #self_labeling_wikicoref_doc()
