import re
import json
import collections
import annotate
import os
import nltk
from mycorenlp import MyCoreNLP


BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")

NER_TYPE_MAP = {"PERSON": "PERSON", "LOCATION": "LOC", "ORGANIZATION": "ORG", "MISC": "", "MONEY": "MONEY",
                "NUMBER": "QUANTITY", "ORDINAL": "ORDINAL", "PERCENT": "PERCENT", "DATE": "DATE",
                "TIME": "TIME", "DURATION": "DATE", "SET": "DATE", "EMAIL": "", "URL": "", "CITY": "GPE",
                "STATE_OR_PROVINCE": "GPE", "COUNTRY": "GPE", "NATIONALITY": "NORP", "RELIGION": "NORP",
                "TITLE": "", "IDEOLOGY": "", "CRIMINAL_CHARGE": "", "CAUSE_OF_DEATH": "",
                "NORP": "NORP", "FAC": "FAC", "ORG": "ORG", "GPE": "GPE", "LOC": "LOC", "PRODUCT": "PRODUCT",
                "EVENT": "EVENT", "WORK_OF_ART": "WORK_OF_ART", "LAW": "LAW", "LANGUAGE": "LANGUAGE",
                "QUANTITY": "QUANTITY", "CARDINAL": "CARDINAL"}


def get_sentences(path):
    docs = []
    for file in os.listdir(path):
        with open(path+"/"+file) as f:
            docs.append(f.read())
    return docs


# write the named entities recognized by stanford corenlp to the record.
def write_ner_coren(nerlist, ner_dict):
    for ner in nerlist:
        if is_valid_ne(ner):
            update_ne_dict_target(ner, ner_dict)


# merge the Spacy result to the record
def integrate_ner(plist, tlist, ner_dict):
    exist_ner_list = ner_dict.keys()
    for ner in plist:
        if is_valid_ne(ner):
            if not check_ner_in_result(ner, exist_ner_list):
                update_ne_dict_target(ner, ner_dict)
            else:
                continue


# check if the ner identified by spacy already exists in the record which stores all the ner identified by standford.
def check_ner_in_result(ner, ner_names):
    #ner_names = [ner[0].lower() for ner in tlist]
    ne_name = ner[0]
    if ner[0].startswith("The "):
        ne_name = ner[0].replace("The ", "the ")
    ner = (ne_name, ner[1])
    # ignore the possessive entity detected by Spacy
    if ner[0][-2:] == "'s":
        return True

    for name in ner_names:
        if annotate.check_two_ner(ner, name):
            return True

    if ner[0] in ner_names or ner[0].strip("the ") in ner_names:
        return True
    elif ner[0].startswith('the '):
        if ner[0][4:] in ner_names or ner[0][4:][:-1] in ner_names:
            return True
    elif ner[0].strip('-').lower() in ner_names:
        return True
    else:
        return False


def merge_ner_target_spacy(ner_list, ner_dict):
    for ner in ner_list:
        if is_valid_ne(ner):
            update_ne_dict_target(ner, ner_dict)


# update named entity record, strip "the " in the entity name.
def update_ne_dict_target(ner, ner_dict):
    ne_name = ner[0]
    if ner[0].startswith("The "):
        ne_name = ner[0].replace("The ", "the ")

    if NER_TYPE_MAP[ner[1]] != "":
        if ne_name not in ner_dict or ne_name.strip("the ") not in ner_dict and NER_TYPE_MAP[ner[1]] != "":
            ner_dict[ne_name] = NER_TYPE_MAP[ner[1]]


def is_valid_ne(ner):
    special_symbol = ['(', ')', '[', '±', ']', '+', '\xa0', '&', '\n', '-RRB-', '-LRB-', '-']
    if ner[0].startswith('A '):
        return False
    if any(sb in ner[0] for sb in special_symbol):
        return False
    if ner[0] == '\n' or ner[0] == ' ':
        return False
    #if ner[1] in ('NUMBER', 'CARDINAL', 'PERCENT', 'ORDINAL', 'DATE', 'TIME', 'MONEY', 'QUANTITY'):
    if ner[1] == 'CARDINAL':
        return False
    else:
        return True


def get_ner_doc(doc):
    sta_nlp = MyCoreNLP()
    ner_token_list = []
    try:
        ner_token_list.extend(sta_nlp.ner(doc))
    except BaseException as e:
        ss = nltk.tokenize.sent_tokenize(doc.replace("\n", ' ').replace('\xa0', ' '))
        part_doc1 = " ".join(ss[:int(len(ss)/2)])
        part_doc2 = " ".join(ss[int(len(ss)/2):])
        ner_token_list.extend(get_ner_doc(part_doc1))
        ner_token_list.extend(get_ner_doc(part_doc2))

    return ner_token_list


def extract_ne_target(path):
    ner_dict = {}
    spa_nlp = annotate.SpacyNLP()
    #sta_nlp = MyCoreNLP()

    for file in os.listdir(path):
        print(file)
        with open(path+"/"+file) as f:
            doc = f.read()
            spa_ner_list = spa_nlp.ents(doc)
            try:
                sta_ner_token_list = get_ner_doc(doc)
            except BaseException as e:
                print(file+" is not able to get stanford named entities")
                continue
            sta_ner_list = annotate.get_merged_ner(sta_ner_token_list)
            write_ner_coren(sta_ner_list, ner_dict)
            integrate_ner(spa_ner_list, sta_ner_list, ner_dict)
            # print(len(set(sta_ner_list)), set(sta_ner_list))
            # merge_ner_target_spacy(set(spa_ner_list), ner_dict)
            # merge_ner_target_stanford(set(sta_ner_list, ner_dict))

    print("number of name entity in target: ", len(ner_dict))

    fn = open('ner_target.txt', 'w+')
    json.dump(sortedDictkeys(ner_dict), fn, indent=4, ensure_ascii=False)
    fn.close()
    return ner_dict


def extract_ne_in_conll(path):
    ne_conll = {}
    # clusters = collections.defaultdict(list)
    coref_stacks = collections.defaultdict(list)
    #coref_stacks = {}
    ne_doc = []
    text = []
    s_id = 0
    with open(path, 'r') as input_file:
        for line in input_file.readlines():
            if line.startswith("#begin document"):
                begin_document_match = re.match(BEGIN_DOCUMENT_REGEX, line)
                doc_id = "{}_{}".format(begin_document_match.group(1), begin_document_match.group(2))
                s_id = 0
            elif line.startswith("#end document"):
                ne_doc = remove_nested_ne(ne_doc)
                update_ne_dict_conll(ne_conll, ne_doc, doc_id)
                ne_doc = []
            else:
                row = line.split()
                if len(row) == 0:
                    text = []
                    s_id += 1
                    continue
                coref = row[-1]
                word_index = row[2]
                token = row[3]
                ner_tag = row[10]
                text.append(token)

                if coref != "-":
                    for segment in coref.split("|"):
                        if segment[0] == "(":
                            if segment[-1] == ")":
                                cluster_id = int(segment[1:-1])
                                #clusters[cluster_id].append((s_id, word_index, word_index, token, ner))
                                #if ner_tag != '*' :
                                ne_doc.append((token, s_id, word_index, word_index, ner_tag, cluster_id))
                                #update_ne_dict_conll(ne_conll, token, doc_id, s_id, word_index, word_index, ner)
                            else:
                                cluster_id = int(segment[1:])
                                coref_stacks[cluster_id].append((word_index, ner_tag))
                        else:
                            cluster_id = int(segment[:-1])
                            start, ner = coref_stacks[cluster_id].pop()
                            #clusters[cluster_id].append((s_id, start, word_index, text[int(start):int(word_index)+1], ner))
                            #if ner != '*':
                            ne_name = " ".join(text[int(start):int(word_index)+1])
                            ne_doc.append((ne_name, s_id, start, word_index, ner, cluster_id))
                            #update_ne_dict_conll(ne_conll, ne_name, doc_id, s_id, start, word_index, ner)

    with open('ner_conll.txt', 'w') as f_conll:
        json.dump(ne_conll, f_conll, indent=4, ensure_ascii=False)

    return ne_conll


def remove_nested_ne(ne_doc):
    #ne_doc.append((ne_name, s_id, start, word_index, ner, cluster_id))
    ne_doc_copy = ne_doc.copy()
    for name, s_id, start, end, ner_tag, coref_id in ne_doc:
        super_ne_list = identify_nested_relation(name, s_id, start, end, coref_id, ne_doc_copy)
        if super_ne_list:
            # need to delete all the annotation of this ne and its super ner
            # or only delete super ner
            #delete_nested_ne_conll(name, ne_doc_copy)
            for super_ne in super_ne_list:
                delete_nested_ne_conll(super_ne[0], ne_doc_copy)
    return ne_doc_copy


def delete_nested_ne_conll(name, ne_in_doc):
    new_ne_doc = ne_in_doc.copy()
    for nm, s_id, start, end, tag, c_id in new_ne_doc:
        if nm == name:
            ne_in_doc.remove((nm, s_id, start, end, tag, c_id))


def identify_nested_relation(ne, s_id, start_index, end_index, c_id, all_ne_doc):
    rst = []
    for ne_name, sid, start, end, ner_tag, coref_id in all_ne_doc:
        # check nested NE in one sentence and in one coreference chain
        if s_id == sid and ne in ne_name:
            if int(start_index) >= int(start) and int(end_index) < int(end):
                rst.append((ne_name, sid, start, end, ner_tag, coref_id))
            elif int(start_index) > int(start) and int(end_index) <= int(end):
                rst.append((ne_name, sid, start, end, ner_tag, coref_id))
            else:
                pass
        if c_id == coref_id and ne in ne_name:
            if len(ne.split()) < len(ne_name.split()):
                rst.append((ne_name, sid, start, end, ner_tag, coref_id))
    return rst


def update_ne_dict_conll(ne_conll, ne_doc, doc_id):
    ne_tmp = {}
    for name, s_id, start, end, ner_tag, c_id in ne_doc:
        if ner_tag == "*" or ner_tag == '*)':
            continue
        if name not in ne_tmp:
            ne_tmp[name] = {'label': ner_tag.replace("*", "").strip("()"),
                            'location': [(doc_id, s_id, start, end)],
                            'substitute': ""}
        else:
            ne_tmp[name]['location'].append((doc_id, s_id, start, end))

    ne_conll[doc_id] = ne_tmp

#def update_ne_dict_conll(ne_dict, name, doc_id, s_id, start_index, end_index, ne_tag):
    # if ne_tag == "*":
    #     return
    # if name not in ne_dict:
    #     ne_dict[name] = {'label': ne_tag.replace("*", '').strip("()"),
    #                      'location': [(doc_id, s_id, start_index, end_index)],
    #                      'substitute': ''}
    # else:
    #     ne_dict[name]['location'].append((doc_id, s_id, start_index, end_index))


def map_ne_target_conll(ne_target, ne_conll):
    ne_target = sortedDictkeys(ne_target)
    ne_conll = sortedDictkeys(ne_conll)
    match_number = 0
    for name, label in ne_target.items():
        replace_flag = False
        for doc_id, info in ne_conll.items():
            if replace_flag:
                break
            for name_conll, value in info.items():
                if value['label'] == label and len(name.split()) == len(name_conll.split()):
                    if value['substitute'] == '':
                        info[name_conll]['substitute'] = name
                        replace_flag = True
                        match_number += 1
                        break
        # for name_conll, value in ne_conll.items():
        #     if value['label'] == label and len(name.split()) == len(name_conll.split()):
        #         if value['substitute'] == '':
        #             #print('find a substitutu for ', name)
        #             ne_conll[name_conll]['substitute'] = name
        #             replace_flag = True
        #             break

        if replace_flag == False:
            print('not found anything for ', name, label)

    map_ne = ne_conll.copy()
    # for doc, info in ne_conll.items():
    #     for k, v in info.items():
    #         if v['substitute'] == "":
    #             del map_ne[doc][k]
    map_ne = sortedDictkeys(map_ne)
    with open('ner_match.txt', 'w') as f_replace:
        json.dump(map_ne, f_replace, indent=4, ensure_ascii=False)
    print("number of matched entity: ", match_number)
    return map_ne


def find_corresponding_ne_in_conll(ne_name, label, ne_conll):
    for name_conll, value in ne_conll:
        if value['label'] == label and len(ne_name.split()) == len(name_conll.split()):
            if value['substitute'] != '':
                ne_conll[name_conll]['substitute'] = ne_name
                break


def sortedDictkeys(dict):
    new_dict = {}
    tmp = sorted(dict.items(), key=lambda obj: obj[0], reverse=True)
    for tmp_item in tmp:
        new_dict[tmp_item[0]] = tmp_item[1]
    return new_dict


def rewrite_conll(path, map_ne):
    close_ner = True
    with open(path) as conll_original_file:
        with open('test_conll_replace.txt', 'w') as conll_new_file:
            for line in conll_original_file.readlines():
                if line.startswith("#begin document"):
                    conll_new_file.write(line)
                    begin_document_match = re.match(BEGIN_DOCUMENT_REGEX, line)
                    doc_id = "{}_{}".format(begin_document_match.group(1), begin_document_match.group(2))
                    s_id = 0
                elif line.startswith("#end document"):
                    conll_new_file.write(line)
                else:
                    row = line.split()
                    if len(row) == 0:
                        s_id += 1
                        conll_new_file.write(line)
                        continue
                    word_index = row[2]
                    ner_tag = row[10]
                    if ner_tag != "*":
                        if ner_tag[0] == "(":
                            if ner_tag[-1] != ")":
                                close_ner = False
                        else:
                            close_ner = True

                    if not close_ner or ner_tag != "*":
                        sub_token = check_replace_token(doc_id, s_id, map_ne, int(word_index))
                        if sub_token is not None:
                            row[3] = sub_token

                    line_str = ["{}".format(element) for element in row]
                    final_line = "  ".join(line_str[i] for i in range(0, len(line_str)))
                    conll_new_file.write(final_line)
                    conll_new_file.write("\n")


def check_replace_token(doc_id, s_id, ne_map, word_index):
    #print(doc_id, s_id, word_index)
    for doc, info in ne_map.items():
        for key, value in info.items():
            #print(key, value)
            for location in value['location']:
                if doc_id == doc and s_id == int(location[1]):
                    #print("doc_id", doc_id)
                    if int(location[2]) <= word_index <= int(location[3]):
                        new_token = value['substitute'].split()[word_index-int(location[2])] \
                                    if value['substitute'] != "" else None
                        return new_token
    return None


if __name__ == '__main__':
    target_path = "manual/"
    conll_path = "all_eng_train_v4_gold_conll"
    #ne_conll = extract_ne_in_conll(conll_path)
    ne_target = extract_ne_target(target_path)

    #with open("ner_target_wikihop.txt") as f_target:
    #    ne_target = json.load(f_target)
    print("number of named entity in target: ", len(ne_target))
    with open('ner_conll_train.txt') as f_conll:
        ne_conll = json.load(f_conll)
    length = 0
    for k, v in ne_conll.items():
        length += len(v)
    print("number of named entity in conll: ", length)
    # with open("ner_match.txt") as fm:
    #     map_ne = json.load(fm)
    map_ne = map_ne_target_conll(ne_target, ne_conll)
    rewrite_conll(conll_path, map_ne)


