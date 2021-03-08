import json
from termcolor import colored
import plotly
import plotly.graph_objs as go
import os
import re


def get_static_one_file_with_possessive(doclines):
    s_id = 0
    ner_dict = {}
    for line in doclines:
        if line.startswith("#"):
            continue
        if line == '\n':
            s_id += 1
        else:
            id_str = line.split()[-1]
            for n in re.finditer("\d+", id_str):
                id = n.group()
                if n.start() > 0 and id_str[n.start()-1] == '(':
                    if id in ner_dict.keys():
                        ner_dict[id]['sId'].append(s_id)
                    else:
                        ner_dict[id] = {'sId': [s_id]}
    #print(ner_dict)
    return ner_dict


def get_statics_one_file(doclines, flag=0):
    coref_chain = 0
    coref_len = 0
    total_distance_para = 0
    ner_details = get_static_one_file_with_possessive(doclines)
    for key, value in ner_details.items():
        if len(value['sId']) > 1:
            coref_chain += 1
            coref_len += len(value['sId'])
            total_distance_ner = 0
            for i in range(1, len(value['sId'])):
                # two ways to compute the distance, one to compute the one with the first appearance
                if flag == 0:
                    distance = value['sId'][i] - value['sId'][0]
                # the other compute the distance between two adjacent appearance
                else:
                    distance = value['sId'][i] - value['sId'][i-1]
                total_distance_ner += distance
            # the average distance inside one coref chain
            aver_dis_ner = int(total_distance_ner / (len(value['sId']) - 1))
            total_distance_para += aver_dis_ner

    print("number of coref chains in one paragraph", coref_chain)
    # this average is not 100% accurate, course some ner with '-' is actually not in the coref_chain
    # but count as one of the chain
    if coref_chain != 0:
        print("average length of coref chains in one paragraph is: ", int(coref_len / coref_chain))
        print("average distance of coref chains in one paragraph is: ", int(total_distance_para / coref_chain))

    return coref_chain, coref_len, total_distance_para


def get_statics(path):
    j = 0
    # number of coref chains for all paragraphs
    coref_chains = []
    aver_coref_lens = []
    aver_total_distance_paras = []
    with open(path) as fc:
        doc = []
        lines = fc.readlines()
        for line in lines:
            if line.startswith("#begin"):
                doc.append(line)
                j += 1
                print(colored(("paragraph " + str(j) + " " + line), 'green'))
            elif line.startswith("#end"):
                doc.append(line)
                chain, len, distance = get_statics_one_file(doc)
                #print(chain, len, distance)
                coref_chains.append(chain)
                if chain != 0:
                    aver_coref_lens.append(int(len/chain))
                    aver_total_distance_paras.append(int(distance/chain))
                else:
                    aver_coref_lens.append(0)
                    aver_total_distance_paras.append(0)
                doc = []
            else:
                doc.append(line)



    count_coref = (coref_chains.count(c) for c in range(max(coref_chains)+1))
    # y should be the number of paragraphs with x coref chains
    x1 = list(range(max(coref_chains)+1))
    y1 = list(count_coref)
    trace1 = go.Bar(
        x=x1,
        y=y1,
        text=y1,
        textposition='outside',
        name="coref chain",
    )
    data1 = [trace1]
    layout1 = go.Layout(
        title="number of coref chains in every document",
        yaxis=dict(title="number of documents"),
        xaxis=dict(title="number of coref chains"),
        barmode='group'
    )
    fig1 = go.Figure(data=data1, layout=layout1)
    plotly.offline.plot(fig1, auto_open=True)

    count_aver_coref_lens = (aver_coref_lens.count(c) for c in range(max(aver_coref_lens)+1))
    x2 = list(range(max(aver_coref_lens)+1))
    y2 = list(count_aver_coref_lens)
    trace2 = go.Bar(
        x=x2,
        y=y2,
        text=y2,
        textposition='outside',
        name="average length of coref chains",
    )
    data2 = [trace2]
    layout2 = go.Layout(
        title="average length of coref chains in every document",
        xaxis=dict(title="length of coref chains"),
        yaxis=dict(title="number of documents"),
        barmode='group'
    )
    fig2 = go.Figure(data=data2, layout=layout2)
    plotly.offline.plot(fig2, auto_open=True)

    count_aver_total_distance_paras = (aver_total_distance_paras.count(c) for c in range(max(aver_total_distance_paras)+1))
    x3 = list(range(max(aver_total_distance_paras)+1))
    y3 = list(count_aver_total_distance_paras)
    trace3 = go.Bar(
        x=x3,
        y=y3,
        text=y3,
        textposition='outside',
        name="average distance of mention paris",
    )
    data3 = [trace3]
    layout3 = go.Layout(
        title="average distance of mention pairs in every document",
        xaxis=dict(title="distance of mention paris"),
        yaxis=dict(title="number of documents"),
        barmode='group'
    )
    fig3 = go.Figure(data=data3, layout=layout3)
    plotly.offline.plot(fig3, auto_open=True)



if __name__ == '__main__':
    get_statics("all_eng_dev_v4_gold_conll")

