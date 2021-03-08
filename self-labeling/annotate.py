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
import math
import multiprocessing
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from mycorenlp import MyCoreNLP

ALLOWED_PARALLEL_PROCESS = 8
MAX_SENTENCES_IN_ONE_DOCUMENT = 30
REMOVE_TAG = "#remove#"
PRONOUNS = {'singular':
                {'female': {'subj': 'she', 'obj': 'her', 'possadj': 'her', 'posspro': 'hers', 'reflx': 'herself'},
                 'male': {'subj': 'he', 'obj': 'him', 'possadj': 'his', 'posspro': 'his', 'reflx': 'himself'},
                 'neutral': {'subj': 'it', 'obj': 'it', 'possadj': 'its', 'posspro': 'its', 'reflx': 'itself'}
                 },
            'plural':
                {'female': {'subj': 'they', 'obj': 'them', 'possadj': 'their', 'posspro': 'theirs',
                            'reflx': 'themselves'},
                 'male': {'subj': 'they', 'obj': 'them', 'possadj': 'their', 'posspro': 'theirs', 'reflx': 'themselves'},
                 'neutral': {'subj': 'they', 'obj': 'them', 'possadj': 'their', 'posspro': 'theirs',
                             'reflx': 'themselves'}
                 }
            }
All_PRONOUNS = ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', \
                'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', \
                'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'themselves']

# the machine which runs StanfordCoreNLPServer
UKP_SERVER = 'http://krusty.ukp.informatik.tu-darmstadt.de'
UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"
LOCALHOST = "http://localhost"


# Use Spacy to identify NER
class SpacyNLP:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def ents(self, text):
        ent_list = []
        doc = self.nlp(text)

        for ent in doc.ents:
            ent_list.append((ent.text, ent.label_))
       # print("Space:", ent_list)
        return ent_list


# given a certain index, find the corresponding dependent parsing result
def find_deparse_with_index(index, deparses):
    index += 1  # deparse result does not have index 0, only 1...length(deparses)
    for deparse in deparses:
        if deparse[2] == index:
            return deparse
        else:
            continue

    return deparses[0]


def chunks(l, n):
    # Yield successive n-sized chunks from l
    for k in range(0, len(l), n):
        yield l[k:k + n]


# get the corresponding pronoun with the given information.
def get_personal_pronoun(deparse_label, gender, number):
    if deparse_label in ('nsubj', 'nsubjpass', 'xsubj'):
        key = 'subj'
    elif deparse_label in ('dobj', 'iobj', 'pobj'):
        key = 'obj'
    elif deparse_label == 'poss':
        key = 'possadj'
    else:  # get the object form of the pronoun by default
        key = 'obj'
    return PRONOUNS[number][gender][key]


# @entity is a tuple like this: ('Florence', 'CITY') or ('Ptolemy', 'PERSON'), or ('Lucy Green', 'PERSON'),
def replace_ner(entity, ner_label, postag, deparse, index):
    gender = 'neutral' if ner_label != 'PERSON' or postag[1] != 'NNP' \
        else GenderRecoginition().gender_identify(entity.upper(), False)
    number = 'plural' if postag[1] == 'NNS' or postag[1] == 'NNPS' else 'singular'
    pronoun = get_personal_pronoun(deparse[0], gender, number)

    if index == 0:
        pronoun = string.capwords(pronoun)
    return pronoun


def update_pronoun(final_result, start_word_id, length, pronoun, ner_ids):
    final_result[start_word_id][3] = pronoun
    final_result[start_word_id][4] = "PRP"
    ner_ids[start_word_id] = ner_ids[start_word_id] + ')' if ner_ids[start_word_id][-1] != ')' else ner_ids[start_word_id]
    for j in range(1, length):
        if ner_ids[start_word_id+j] != '-' and ner_ids[start_word_id+j] != ner_ids[start_word_id]:
            ner_ids[start_word_id] = ner_ids[start_word_id] + '|' + str(ner_ids[start_word_id+j])
        final_result[start_word_id+j][3] = REMOVE_TAG

    #ner_ids[start_word_id] = ner_ids[start_word_id].replace('-', '({}'.format(ner_id)) \
       # if ner_ids[start_word_id] == '-' else ner_ids[start_word_id] + '|(' + str(ner_id)


def generate_conll_style_output(sentence, sid, doc_id, part, final_result, dcoref):
    sta_nlp = MyCoreNLP()
    # part: _part 002 or _part 011
    part_number = int(part[7:])
    tokens = sta_nlp.word_tokenize(sentence)
    poss = sta_nlp.pos_tag(sentence)
    assert(len(tokens) == len(poss))
    ner_ids = ['-']*len(tokens)

    for i in range(len(tokens)):
        # stanford word tokenize do not identify "1 1/2" as two token but one token
        token = tokens[i].replace(" ", "-")
        pos = poss[i][1]
        line_element = [doc_id, part_number, '-', token, pos, '*', '-', '-', '-', '-', '*', '-']
        final_result.append(line_element)

    for id, mentions in dcoref.items():
        # a mention is like this (2, 1, 3, 'The city'), sentence id starts from 1,
        for mention in mentions:
            if mention[0] == sid + 1:
                update_ner_id(ner_ids, mention[1]-1, mention[2]-mention[1], id, mention[-1], sid, dcoref)

    return ner_ids


def dcoref_record(doc, doc_id):
    sta_nlp = MyCoreNLP()
    coref_id = 0
    dcoref_dict = {}
    try:
        #dcoref = sta_nlp.dcoref(" ".join(ss))
        dcoref = sta_nlp.dcoref(doc)
    except BaseException as e:
        print("Time out when call stanford dcoref annotator in ", doc_id)
        #print(doc)
        #raise e
    else:
        for coref_chain in dcoref:
            dcoref_dict[coref_id] = coref_chain
            coref_id += 1

    #print("dcoref:", dcoref_dict)
    return dcoref_dict


# generate an output CoNLL file for every document.
def output(ss, dict_n, doc_id, part, file_path, doc):
    # if ss is empty, simply return
    if not ss:
        return
    fo = open(file_path, 'w')
    fo.write("#begin document ("+doc_id+"); "+part[1:])
    annotated_ner = []
    replaced_ner = []
    np_id = {"id": len(dict_n)}
    dcoref = dcoref_record(doc, doc_id+part)
    for sentence in ss:
        try:
            # this assumes that there will not be two identical sentences
            if sentence == " ":
                continue
            final_result = []
            ner_ids = generate_conll_style_output(sentence, ss.index(sentence), doc_id, part, final_result, dcoref)
            self_labeling(sentence, ss.index(sentence), dict_n, annotated_ner, np_id, replaced_ner, final_result, ner_ids, dcoref, fo)
        except BaseException as e:
            print(colored(e, 'green'))
            print(colored(("found exception in sentence: " + sentence, "should not happen very offen"), 'green'))
            #raise e
        else:
            continue
    fo.write("\n#end document")
    fo.close()


# get all the named entities in a given sentence
def get_ner_in_sentence(sid, dict_n):
    ner_in_sentence = {}
    for key, value in dict_n.items():
        if sid in value['sId']:
            ner_in_sentence[key] = value
    return ner_in_sentence


def identity_ner_in_dcoref(ner, start_index, sid, dcoref):
    for coref_id, mentions in dcoref.items():
        for mention in mentions:
            # pronoun and named entity should be treated differently
            if mention[-1].lower() in All_PRONOUNS:
                if ner.lower() == mention[-1].lower() and sid == mention[0]-1:
                    if start_index == mention[1] - 1:
                        return coref_id
            else:
                #if ner.lower() == mention[-1].lower():
                if ner.lower().strip("the ") == mention[-1].lower().strip("the ") or \
                        ner.lower().strip("the ") == mention[-1].lower()[:-2]:
                    return coref_id
    return None


# add the coreference annotation, i.e., ner id
def update_ner_id(ner_ids, start_word_id, length, ner_id, ner, sid, dcoref, replace_flag=False):
    # check if this named entity is already annotated in dcoref
    new_id = identity_ner_in_dcoref(ner, start_word_id, sid, dcoref)
    ner_id = new_id if new_id is not None else ner_id + len(dcoref)

    end_word_id = start_word_id + length - 1
    # to deal with the case that sentence in dcoref is longer than the existing one
    if start_word_id > len(ner_ids) - 1 or end_word_id > len(ner_ids) - 1:
        return

    # check if there already exists an annotation for this mention
    if ner_ids[start_word_id] != '-' and ner_ids[end_word_id] != '-':
        if any(ner_id == int(old_id) for old_id in re.findall("\d+", ner_ids[start_word_id])) and \
                any(ner_id == int(old_id) for old_id in re.findall("\d+", ner_ids[end_word_id])):
            return

    if ner_ids[start_word_id] == '-':
        ner_ids[start_word_id] = ner_ids[start_word_id].replace('-', '({}'.format(ner_id))
    else:
        #if not any(ner_id == int(old_id) for old_id in re.findall("\d+", ner_ids[start_word_id])):
        ner_ids[start_word_id] = ner_ids[start_word_id] + '|(' + str(ner_id)

    if not replace_flag:
        if end_word_id != start_word_id:
            if ner_ids[end_word_id] == '-':
                ner_ids[end_word_id] = ner_ids[end_word_id].replace('-', '{})'.format(ner_id))
            else:
                #if not any(ner_id == int(old_id) for old_id in re.findall("\d+", ner_ids[end_word_id])):
                ner_ids[end_word_id] = ner_ids[end_word_id] + '|' + str(ner_id) + ')'
        else:
            ner_ids[end_word_id] = str(ner_ids[end_word_id]) + ')'


# replace existing possessive entity appearing more than once with possessive pronoun.
def update_possesive(result, dict_n, sid, replaced_flag):
    for i in range(len(result)):
        if result[i][3] == "'s" and result[i][-1] == "-":
            former_index = i - 1
            former = result[former_index][3]
            if former.lower() == 'he' or former.lower() == 'him':
                pronoun = 'his'
            elif former.lower() == 'she' or former.lower() == 'her':
                pronoun = 'her'
            elif former.lower() == 'it':
                pronoun = 'its'
            # if this entity is the first time appear, may result in a pronoun before mention
            elif former.lower() in dict_n and sid != dict_n[former.lower()]["sId"][0] and not replaced_flag and \
                    len(dict_n[former.lower()]["sId"]) > 1 and result[former_index][-1] == '-':
                label = dict_n[former.lower()]["label"]
                gender = 'neutral' if label != 'PERSON' else GenderRecoginition().gender_identify(former.upper(), False)
                pronoun = get_personal_pronoun("poss", gender, "singular")
                result[former_index][-1] = "("+str(dict_n[former.lower()]["nerId"])+")"
            else:
                continue
            result[i][3] = REMOVE_TAG
            result[former_index][4] = "PRP$"
            replaced_flag = True

            # if this entity is the first word of the sentence, the first letter of pronoun should be capital
            if former_index == 0:
                pronoun = string.capwords(pronoun)
            else:
                pronoun = remove_the_before_pronoun(result, former_index, pronoun)

            result[former_index][3] = pronoun


def remove_the_before_pronoun(result, pronoun_index, pronoun):
            # the lowercase it ensures this "it" wont be the first token, so former_index wont be less than zero
        if pronoun_index > 1:
            former_index = pronoun_index - 1
            former = result[former_index][3]
            if former.lower() == "the":
                result[former_index][3] = REMOVE_TAG
                # if ner id of token 'the' is not '-', then it may be something like this (3
                ner_id = result[former_index][-1]
                if ner_id != '-':
                    result[pronoun_index][-1] = ner_id if result[pronoun_index][-1] == '-' else result[pronoun_index][-1] + '|' + ner_id
                # if former token "the" is the first word of the sentence, the first letter of pronoun should be capital
                if former_index == 0:
                    pronoun = string.capwords(pronoun)
        return pronoun


def identity_nested_relation_dcoref(ner, srt, lth, sid, dcoref):
    for mentions in dcoref.values():
        for mention in mentions:
            if sid == mention[0]-1 and ner in mention[-1]:
                if srt >= mention[1]-1 and (srt+lth) < mention[2]-1:
                    return mention[-1]
                elif srt > mention[0]-1 and (srt+lth) <= mention[2]-1:
                    return mention[-1]
                else:
                    pass
    return None


# check if some ne is nested inside other longer named entities, eg: "Central European" in "Central European Time"
def identify_nested_relation(ne, srt, lth, ner_in_sentence_pos):
    for start, length, ner, value in ner_in_sentence_pos:
        if ne in ner:
            if srt >= start and (srt+lth) < (start+length):
                return [(start, length, ner, value)]
            elif srt > start and (srt+lth) <= (start+length):
                return [(start, length, ner, value)]
            else:
                pass
    return None


# get all the named entities that appear more than once in a sentence
def get_ner_to_mark(exist_ner_with_pos):
    ner_to_mark = []
    for start, length, ner, value in exist_ner_with_pos:
        # if this ner appears only once, no need to annotation
        if len(value['sId']) <= 1:
            continue
        # if this ner appears more than once, need to mark or replace with pronoun
        else:
            ner_to_mark.append((start, length, ner, value))

    return ner_to_mark


# a helper function for find_possessive_pronoun
def add_result(rst_dict, start, np_len, possessive_idx):
    if start not in rst_dict.keys():
        rst_dict[start] = {"len": np_len, "prp_indices": [possessive_idx]}
    else:
        rst_dict[start]["prp_indices"].append(possessive_idx)


# given the possessive pronoun and possible noun phrase tree, check whether PRP$ and np match
def check_possessive_pronoun_number(possprn, np_tree):
    head_word = find_head_word(np_tree)
    if head_word is not None:
        index = np_tree.leaves().index(head_word[0])
    else:
        index = len(np_tree.leaves()) - 1
        head_word = [np_tree.leaves()[-1]]

    head_position = np_tree.leaf_treeposition(index)
    head_tree = np_tree[head_position[:-1]]
    # if the noun phrase is a pronoun, donot mark
    if head_tree.label() == "PRP":
        return False

    if head_tree.label() == "NNS" or head_tree.label() == "NNPS":
        if possprn == "their":
            return True
    else:
        if possprn in PRONOUNS['singular']['male'].values():
            if GenderRecoginition().gender_identify(head_word[0].upper(), False) == "male":
                return True
        elif possprn in PRONOUNS['singular']['female'].values():
            if GenderRecoginition().gender_identify(head_word[0].upper(), False) == "female":
                return True
        else:
            if possprn != "their":
                return True

    return False


# find possessive pronouns and the matching noun phrases step2
def find_possessive_pronoun(tree, dparses, rst_dict, ner_ids, clause_idx=0, subj_idx=0):
    subj_token_index = -1
    poss_indices = []
    if len(tree.leaves()) != len(dparses):
        return None

    for dparse in dparses:
        if dparse[0] in ('nsubj', 'nsubjpass', 'xsubj'):
            # dependency parse starts from 1
            subj_token_index = dparse[2] - 1
            break

    for dparse in dparses:
        if dparse[0] in ("nmod:poss", "poss"):
            possessive_index = dparse[2] - 1
            # there may be multiple possesive pronouns in one sentence, should find all of them.
            if possessive_index > 0 and ner_ids[possessive_index] == '-':
                poss_indices.append(possessive_index)

    leaves = tree.leaves()
    for possessive_index in poss_indices:
        possessive_position = tree.leaf_treeposition(possessive_index)
        possessive_pronoun = tree.leaves()[possessive_index]
        prp_tree = tree[possessive_position[:-1]]
        if prp_tree.label() == "PRP$":
            offset = 1
            if leaves[possessive_index-1] == "and" and possessive_index >= 2:
                left_position = tree.leaf_treeposition(possessive_index - 2)
                if tree.leaves()[possessive_index-2] == ',':
                    np_tree = tree[left_position[:-1]].left_sibling()
                    offset += 1
                else:
                    np_tree = tree[left_position[:-2]]

                if np_tree.label() == "NP":
                    np = np_tree.leaves()
                    start = possessive_index - len(np) - offset + clause_idx + subj_idx
                    if start in range(0, possessive_index):
                        if check_possessive_pronoun_number(possessive_pronoun, np_tree):
                            add_result(rst_dict, start, len(np), possessive_index+clause_idx+subj_idx)

                # and_position = tree.leaf_treeposition(possessive_index-1)
                # left = tree[and_position[:-1]].left_sibling()
                # if left is not None and left.label() == "NP":
                #     np = left.leaves()
                #     start = possessive_index - len(np) - 1 + clause_idx + subj_idx
                #     add_result(rst_dict, start, len(np), possessive_index+clause_idx+subj_idx)

            elif subj_token_index != -1 and subj_token_index < possessive_index:
                subj_position = tree.leaf_treeposition(subj_token_index)
                # find the nearest NP of subject
                subj_tree = tree[subj_position[:-2]]
                if subj_tree.label() == "NP":
                    np = subj_tree.leaves()
                    # if the subject token repeats in the noun phrase, there may be a wrong annotation
                    for i in range(len(np)):
                        if np[i] == tree.leaves()[subj_token_index]:
                            start = subj_token_index - i + clause_idx + subj_idx
                            break
                    # solved the problem when prp$ not part of a VP
                    #vb_tree = tree[possessive_position[:2]]
                    #if vb_tree.label() == "VP":
                    if check_possessive_pronoun_number(possessive_pronoun, subj_tree):
                        add_result(rst_dict, start, len(np), possessive_index+clause_idx+subj_idx)
            else:
                break
    return rst_dict


# find possessive pronouns and the matching noun phrases step1
def mark_possesive_pronoun(tree, dparses, nlp, ner_ids):
    rst_dict = {}
    clause_start_index, clause_len = find_clause(tree, nlp)
    # if may be that the whole sentence is a SBAR clause
    if clause_start_index == -1 or clause_start_index == 0 and clause_len == len(tree.leaves()):
        find_possessive_pronoun(tree, dparses, rst_dict, ner_ids)
    else:
        # clause sentence is in the beginning of a sentence
        if clause_start_index == 0:
            subj_start_idx = clause_len
            subj = TreebankWordDetokenizer().detokenize(tree.leaves()[clause_len:])
            clause = TreebankWordDetokenizer().detokenize(tree.leaves()[:clause_len])
        else:
            subj_start_idx = 0
            subj = TreebankWordDetokenizer().detokenize(tree.leaves()[:clause_start_index])
            clause = TreebankWordDetokenizer().detokenize(tree.leaves()[clause_start_index:])

        subj_tree = nltk.tree.ParentedTree.fromstring(nlp.parse(subj))
        clause_tree = nltk.tree.ParentedTree.fromstring(nlp.parse(clause))

        find_possessive_pronoun(subj_tree, nlp.dependency_parse(subj), rst_dict, ner_ids, 0, subj_start_idx)
        find_possessive_pronoun(clause_tree, nlp.dependency_parse(clause), rst_dict, ner_ids, clause_start_index, 0)

    return rst_dict


# find clause of a sentence
def find_clause(tree, nlp):
    clause = "***###***"
    clause_idx_str = -1
    # list to string
    str_leaves = " ".join(tree.leaves())
    for node in tree.subtrees(lambda x: x.label() == "SBAR"):
        clause = node.leaves()
        clause_idx_str = str_leaves.find(" ".join(clause))
    if clause_idx_str != -1:
        # string to token list
        clause_start_index = nlp.word_tokenize(str_leaves[:clause_idx_str])
        # clause start index, eg: if len(sub_sentence)=8, then leaves[8:] is the clause leaves
        return len(clause_start_index), len(clause)
    else:
        return -1, 0


# find the head word of a noun phrase
def find_head_word(ne_tree):
    for sub_tree in reversed(ne_tree):
        if isinstance(sub_tree, nltk.tree.Tree):
            if sub_tree.label() in ['NP', 'NN', 'NNS', 'NNP', 'NNPS', 'FRAG']:
                return find_head_word(sub_tree)
        else:
            return ne_tree.leaves()


# find the index of the head word in a noun phrase
def find_head_word_index(start, length, ne_tree):
    head_word = find_head_word(ne_tree)
    if head_word is not None:
        head_word_index = start + ne_tree.leaves().index(head_word[0])
    # if cannot find a head word,use the rightmost word instead.
    else:
        head_word_index = start + length - 1
    return head_word_index


# the key function for processing a sentence
def self_labeling(sentence, sid, dict_n, annotated_ner, np_id, replaced_ner, final_result, ner_ids, dcoref, fp):
    replace_flag = False
    sta_nlp = MyCoreNLP()
    # potential error: "32 1/5" will be considered as one token with a space, should be considered as two tokens
    tokens = sta_nlp.word_tokenize(sentence)
    poss = sta_nlp.pos_tag(sentence)
    # assert(len(tokens) == len(poss))
    # the nlp parse annotator can only handle a sentence with less than 80 tokens
    if len(tokens) >= 75:
        return
    # solve the problem of sentence starts with "
    cparses = sta_nlp.parse(sentence.replace("\"", "'"))
    dparses = sta_nlp.dependency_parse(sentence)
    # ner_ids = ['-']*len(tokens)
    # final_result = []
    # for i in range(len(tokens)):
    #     token = tokens[i]
    #     pos = poss[i][1]
    #     line_element = [doc_id, part_number, '-', token, pos, '*', '-', '-', '-', '-', '*', '-']
    #     final_result.append(line_element)

    # get all the ner that appears in this sentence from ner_dict map
    ner_in_sentence = get_ner_in_sentence(sid, dict_n)
    exist_ner_with_pos = []
    for ner, value in ner_in_sentence.items():
        ner_tokens = sta_nlp.word_tokenize(ner)
        # find all the ner appeared in one sentence
        for n in re.finditer(ner.replace(".", "\."), sentence.lower()):
            # solve the problem of tokenize difference for concatenate words like Austria-Hungary or  GMT/UTC.
            if n.start() > 0 and sentence[n.start()-1] not in (' ', '(', '"', '\''):
                continue
            # " ' " should also be considered as end of a word, like "Lucy's"
            if n.end() < len(sentence) and sentence[n.end()] not in (' ', '.', ',', ':', '"', '?', ')', '\'', '!'):
                continue

            start_index = len(sta_nlp.word_tokenize(sentence[:n.start()]))
            ne_len = len(ner_tokens)
            # added on 20190408, since all the ne in the record do not contain the, it is necessary to add here.
            if start_index > 0 and tokens[start_index-1].lower() == "the":
                start_index = start_index - 1
                ne_len = len(ner_tokens) + 1
                ner = "the "+ner
                ner_tokens_with_the = ['the']
                ner_tokens_with_the.extend(ner_tokens)
                ner_tokens = ner_tokens_with_the

            # to check whether the found start index and len correspond to the ne
            if " ".join(ner_tokens).lower() == " ".join(tokens[start_index:start_index+ne_len]).lower():
                exist_ner_with_pos.append((start_index, ne_len, ner, value))

    # identify nested ner
    exist_ner_with_pos_copy = copy.deepcopy(exist_ner_with_pos)
    ner_to_mark = get_ner_to_mark(exist_ner_with_pos_copy)

    # add annotation for exsiting possessive pronouns
    tree = nltk.tree.ParentedTree.fromstring(cparses)
    possessive_pronoun = mark_possesive_pronoun(tree, dparses, sta_nlp, ner_ids)
    # if there is no annotation in this sentence, skip
    # if len(ner_to_mark) == 0 and possessive_pronoun == {}:
    #     return

    for start, value in possessive_pronoun.items():
        np_id["id"] = np_id["id"] + 1
        tmp_id = np_id["id"]
        # check if this noun phrase is also a named entity
        np = " ".join(tokens[start:start+value["len"]]).lower()
        # since all the ne in dict do not have "the ", need to consider that
        np_key = np[4:] if np.startswith("the ") else np
        if np_key in dict_n.keys():
            tmp_id = dict_n[np_key]["nerId"]
            # if this np is a named entity and already appears more than once, it will be annotated later
            if len(dict_n[np_key]["sId"]) <= 1:
                update_ner_id(ner_ids, start, value["len"], tmp_id, np, sid, dcoref)
        else:
            update_ner_id(ner_ids, start, value["len"], tmp_id, np, sid, dcoref)

        for prp_index in value["prp_indices"]:
            update_ner_id(ner_ids, prp_index, 1, tmp_id, np, sid, dcoref)

    ner_to_replace_with_pos = []
    # sort the ner based in descending order of ner length
    ner_to_mark = sorted(ner_to_mark, key=lambda x: x[1], reverse=True)
    for start, length, ner, value in ner_to_mark:
        super_ner = identify_nested_relation(ner, start, length, exist_ner_with_pos_copy)
        # need to check if this ne is a sub entity in the dcoref
        super_ner_in_docref = identity_nested_relation_dcoref(ner, start, length, sid, dcoref)
        # if this ner appears more than once, this is the first time
        if ner not in annotated_ner:
            update_ner_id(ner_ids, start, length, value['nerId'], ner, sid, dcoref)
            annotated_ner.append(ner)
        else:
            # need to check if this ne is a sub entity in the dcoref
            if super_ner is None and super_ner_in_docref is None:
                # find all the ner need to be replaced in one sentence
                ner_to_replace_with_pos.append((start, length, ner, value))

    # if there are ners to be replaced
    if len(ner_to_replace_with_pos) == 0:
        pass
    else:
        start, length, ner, value = ner_to_replace_with_pos[0]
        ne_tree = nltk.tree.ParentedTree.fromstring(sta_nlp.parse(ner))
        deparse_index = find_head_word_index(start, length, ne_tree)
        deparse = find_deparse_with_index(deparse_index, dparses)
        pronoun = replace_ner(ner, value['label'], poss[deparse_index], deparse, start)

        # check if the ne be replaced has already been marked
        #if any(ner_ids[j] != '-' for j in range(start, start+length)):
        #  check if the ne be replaced has already been marked and replace only the nearest ner with pronoun
        if any(ner_ids[j] != '-' for j in range(start, start+length)) or ner in replaced_ner:
            update_ner_id(ner_ids, start, length, value['nerId'], ner, sid, dcoref)
        else:
            update_ner_id(ner_ids, start, length, value['nerId'], ner, sid, dcoref, True)
            update_pronoun(final_result, start, length, pronoun, ner_ids)
            replaced_ner.append(ner)
            replace_flag = True

        # the second ne need to be replaced will only be annotated, not replaced
        if len(ner_to_replace_with_pos) > 1:
            for j in range(1, len(ner_to_replace_with_pos)):
                start, length, ner, value = ner_to_replace_with_pos[j]
                if not any(final_result[k][3] == REMOVE_TAG for k in range(start, start+length)):
                    update_ner_id(ner_ids, start, length, value['nerId'], ner, sid, dcoref)

    write_final_result(final_result, ner_ids, dict_n, sid, replace_flag, fp)


def write_final_result(final_result, ids, dict_n, sid, flag, fp):
    fp.write("\n")

    for k in range(len(final_result)):
        final_result[k][-1] = ids[k]

    # this step is very necessary
    final_result = [line for line in final_result if line[3] != REMOVE_TAG]
    update_possesive(final_result, dict_n, sid, flag)
    final_result = [line for line in final_result if line[3] != REMOVE_TAG]
    merge_dt_and_ne(final_result)

    for n in range(len(final_result)):
        final_result[n][2] = n
    for line_elem in final_result:
        line_str = ["{}".format(element) for element in line_elem]
        line_width = [20, 5, 5, 30, 5, 5, 5, 5, 5, 5, 5, 5, 10]
        line = "  ".join(line_str[i].rjust(line_width[i]) for i in range(0, len(line_str)))
        fp.write(line)
        fp.write("\n")


def merge_dt_and_ne(results):
    for i in range(len(results)):
        if i == len(results) - 1:
            continue
        if results[i][3].lower() == 'the' and results[i][-1] != '-':
            latter_index = i + 1
            if results[latter_index][-1] != '-':
                for id in re.findall("\d+", results[i][-1]):
                    if "("+id in results[i][-1] and "("+id in results[latter_index][-1]:
                        end_index = find_the_end_of_ne(results, i, id)
                        if id+")|"+id+")" not in results[end_index][-1]:
                            continue
                        if latter_index == end_index:
                            results[latter_index][-1] = results[latter_index][-1].replace("("+id+")", "")
                        else:
                            # fix bug caused by strip("("+id)
                            results[latter_index][-1] = results[latter_index][-1].replace("("+id, "")
                            results[end_index][-1] = results[end_index][-1].replace(id+")|"+id+")", id+")", 1)
                        results[latter_index][-1] = check_split_line_in_nerid(results[latter_index][-1])
                        results[end_index][-1] = check_split_line_in_nerid(results[end_index][-1])


def find_the_end_of_ne(results, index, id):
    for i in range(index, len(results)):
        if id+")" in results[i][-1]:
            return i


def check_split_line_in_nerid(id):
    new_id = id.replace("||", "|")
    if id.startswith("|"):
        new_id = new_id[1:]
    if id == '':
        new_id = '-'
    elif id[-1] == "|":
        new_id = new_id[:-1]
    else:
        pass

    return new_id


# merge the tokens in standford corenlp ner result to named entities.
def get_merged_ner(orig_list):
    merged_ner_list = []
    for i in range(len(orig_list)):
        en = orig_list[i]
        current_index = i
        # merge entity_list next to each other with the same NER
        if en[1] != 'O':
            merged_ner_list.append(en)
            if current_index > 0:
                former_ne = orig_list[current_index - 1]
                if en[1] == former_ne[1]:
                    orig_list[current_index] = (former_ne[0] + ' ' + en[0], en[1])
                    merged_ner_list.append(orig_list[current_index])
                    merged_ner_list.remove(en)
                    merged_ner_list.remove(former_ne)
                # added on 20190114, solve problem caused by "the it" begin
                elif former_ne[0].lower() == "the":
                    orig_list[current_index] = ("the" + ' ' + en[0], en[1])
                    merged_ner_list.append(orig_list[current_index])
                    merged_ner_list.remove(en)
                else:
                    continue
                # end
   # print("stanford:", merged_ner_list)
    return merged_ner_list


# get all the sentences in a paragraph
def get_sentences(paragraph):
    all_context = []
    # parse json style data, get the context field, split each context into sentences, and process each sentences
    for paraph in paragraph:
        context = paraph['context']
        all_context.append(context)

    ss_in_context = " ".join(all_context).replace('\n', ' ').replace('\xa0', ' ')
        #ss.extend(nltk.tokenize.sent_tokenize(ss_in_context))
    #return nlp.ssplit(ss_in_context)
    return ss_in_context
    #ss.extend(nlp.ssplit(ss_in_context))
    #return ss


# extract all named entities in a document with given sentences.
def extract_ner(sentences, ner_dict, title, path):
    sta_nlp = MyCoreNLP()
    spa_nlp = SpacyNLP()

    for j in range(len(sentences)):
        sentence_id = j
        sentence = sentences[j]
       # print(sentence_id, sentence)
        # get ner identified by stanford corenlp
        sta_ner_list = sta_nlp.ner(sentence)
        # get ner identified by spacy
        spa_ner_list = spa_nlp.ents(sentence)
        merged_ner_coren = get_merged_ner(sta_ner_list)
        write_ner_coren(merged_ner_coren, sentence_id, ner_dict)
        integrate_ner(spa_ner_list, merged_ner_coren, sentence_id, ner_dict)

    fn = open(path + '/' + title + '.txt', 'a+')
    json.dump(ner_dict, fn, indent=4, ensure_ascii=False)
    fn.close()
    return ner_dict


# ner: identified by spacy, name: ner name already identified by CoreNLP
def check_two_ner(ner, name):
    # added on 20190114 "the Navy" and "Navy" should be identified as one ne begin
    if ner[0].lower() in name:
        return True
    # end

    i = 0
    for word in ner[0].lower().split(' '):
        if word in name:
            i += 1
    return True if i > 3 else False


# check if the ner identified by spacy already exists in the record which stores all the ner identified by standford.
def check_ner_in_result(ner, tlist):
    ner_names = [ner[0].lower() for ner in tlist]
    # ignore the possessive entity detected by Spacy
    if ner[0][-2:] == "'s":
        return True

    for name in ner_names:
        if check_two_ner(ner, name):
            return True

    if ner[0].lower() in ner_names or ner[0].strip("the ") in ner_names:
        return True

    elif ner[0].lower().startswith('the '):
        if ner[0][4:].lower() in ner_names or ner[0][4:][:-1].lower() in ner_names:
            return True
    elif ner[0].strip('-').lower() in ner_names:
        return True
    else:
        return False


# merge the Spacy result to the record
def integrate_ner(plist, tlist, sid, ner_dict):
    for ner in plist:
        if is_valid_ne(ner):
            if not check_ner_in_result(ner, tlist):
                update_ner_dict(ner, sid, ner_dict)
            else:
                continue


# update named entity record, strip "the " in the entity name.
def update_ner_dict(ner, sid, ner_dict):
    ner_id = len(ner_dict)
    # there may be two "the" in one NE, so simply strip("the ") is not enough.
    ne_key = ner[0].lower()[4:] if ner[0].lower().startswith("the ") else ner[0].lower()
    #if ner[0].lower() in ner_dict:
    if ne_key in ner_dict:
        ner_dict[ne_key]['sId'].append(sid)
        #ner_dict[ner[0].lower()]['sId'].append(sid)
    else:
        ner_dict[ne_key] = {'label': ner[1], 'sId': [sid], 'nerId': ner_id}
        #ner_dict[ner[0].lower()] = {'label': ner[1], 'sId': [sid], 'nerId': ner_id}


# check if a given named entity will be considered or not.
def is_valid_ne(ner):
    special_symbol = ['(', ')', '[', '±', ']', '+', '\xa0', '&']
    if ner[0].startswith('A '):
        return False
    if any(sb in ner[0] for sb in special_symbol):
        return False
    if ner[0] == '\n' or ner[0] == ' ':
        return False
    if ner[1] in ('NUMBER', 'CARDINAL', 'NATIONALITY', 'PERCENT', 'ORDINAL', 'DATE', 'DURATION', 'SET'):
        return False
    else:
        return True


# write the named entities recognized by stanford corenlp to the record.
def write_ner_coren(nerlist, sid, ner_dict):
    for ner in nerlist:
        if is_valid_ne(ner):
            update_ner_dict(ner, sid, ner_dict)


def paragraph_process(sentences, paraph_name, path):
    # split one paragraph into several parts
    if len(sentences) > MAX_SENTENCES_IN_ONE_DOCUMENT:
        parts_number = math.ceil(len(sentences) / MAX_SENTENCES_IN_ONE_DOCUMENT)
        for k in range(int(parts_number)):
            if k < 10:
                part = "_part 00" + str(k)
            else:
                part = "_part 0" + str(k)
            part_sentences = sentences[(MAX_SENTENCES_IN_ONE_DOCUMENT*k):(MAX_SENTENCES_IN_ONE_DOCUMENT*(k+1))]
            document_process(part_sentences, paraph_name, part, path, " ".join(part_sentences))
    else:
        document_process(sentences, paraph_name, "_part 000", path, " ".join(sentences))


def document_process(sentences, paraph_name, part, path, doc):
    ner_dict = {}
    ner_dict = extract_ner(sentences, ner_dict, paraph_name + part, path)
    doc_id = path + '/' + paraph_name.replace(" ", "/00/")
    file_path = path + '/' + paraph_name + part + "_annotate.txt"
    output(sentences, ner_dict, doc_id, part, file_path, doc)


def annotate_document(file_name, doc, output_path, doc_index):
    print(colored(("paragraph " + str(doc_index) + " " + file_name), 'yellow'))
    nlp = MyCoreNLP()
    ss = []
    try:
        sentences = nlp.ssplit(doc)
    except BaseException as e:
        ss.extend(nltk.tokenize.sent_tokenize(doc.replace("\n", ' ').replace('\xa0', ' ')))
        print("The document "+file_name+" is too long, will be split automatically.")
        #raise e

    if not ss:
        paragraph_process(sentences, file_name, output_path)
    # if the document has too many sentences, then split into two documents
    else:
        part_doc1 = " ".join(ss[:int(len(ss)/2)])
        part_doc2 = " ".join(ss[int(len(ss)/2):])
        annotate_document(file_name, part_doc1, output_path, doc_index)
        annotate_document(file_name+"1", part_doc2, output_path, doc_index)


if __name__ == '__main__':
    doc_path = sys.argv[1]
    output_path = sys.argv[2]
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    pool = multiprocessing.Pool(processes=ALLOWED_PARALLEL_PROCESS)
    i = 0
    for file in os.listdir(doc_path):
        i += 1
        with open(doc_path+'/'+file) as f:
            doc = f.read()
            filename = file[:-4] if file.endswith(".txt") else file
            #annotate_document(filename, doc, output_path, i)
            pool.apply_async(annotate_document, (filename, doc, output_path, i, ))
    pool.close()
    pool.join()
    print("the end!")

    # sentences = []
    # ner_dict = {}
    # nlp = StanfordNLP()
    # context="Most locations used a 32 1/5 ft ( 9.8 meters ) - diameter version that straddles the building and is aimed at the intersection ."
    # sentences = nlp.ssplit(context)
    #sentences.extend(nltk.tokenize.sent_tokenize(context.replace("\n", ' ').replace('\xa0', ' ')))
    # print(nlp.dcoref(context))
    # for s in sentences:
    #     print(nlp.pos(s))
    #     print(nlp.parse(s))
    #    tree = nltk.tree.ParentedTree.fromstring(nlp.parse(s))
    #     tree.pretty_print()
    #     print(find_head_word(tree))
    #     print(nlp.dependency_parse(s))
    # #ner_dict = extract_ner(sentences, ner_dict, 'input_part 000', path)
    # file_path = path + '/' +  "input_part 000_annotate.txt"
    # output(sentences, ner_dict, 'input', '_part 000', file_path)

