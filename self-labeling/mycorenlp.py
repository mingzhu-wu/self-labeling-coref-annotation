from stanfordcorenlp import StanfordCoreNLP
import logging
import sys
import json
import requests
from nltk.tokenize.treebank import TreebankWordDetokenizer

# the machine which runs StanfordCoreNLPServer
UKP_SERVER = 'http://krusty.ukp.informatik.tu-darmstadt.de'
UKP_SERVER_NED = "http://ned.ukp.informatik.tu-darmstadt.de"
LOCALHOST = "http://localhost"


class MyCoreNLP(StanfordCoreNLP):

    def __init__(self, host=UKP_SERVER, port=9000, timeout=1500):
        StanfordCoreNLP.__init__(self, host, port, timeout=timeout)

    def _dcoref_request(self, annotators=None, data=None, *args, **kwargs):
        if sys.version_info.major >= 3:
            data = data.encode('utf-8')

        properties = {'annotators': annotators, 'timeout': 150000, 'outputFormat': 'json', \
                      'dcoref.sievePasses': 'MarkRole,DiscourseMatch,ExactStringMatch,RelaxedExactStringMatch,PreciseConstructs',\
                      'dcoref.postprocessing': 'true'}
        params = {'properties': str(properties), 'pipelineLanguage': self.lang}
        if 'pattern' in kwargs:
            params = {"pattern": kwargs['pattern'], 'properties': str(properties), 'pipelineLanguage': self.lang}

        logging.info(params)
        r = requests.post(self.url, params=params, data=data, headers={'Connection': 'close'})
        r_dict = json.loads(r.text)

        return r_dict

    def dcoref(self, text):
        r_dict = self._dcoref_request('dcoref', text)
        corefs = []
        for k, mentions in r_dict['corefs'].items():
            if len(mentions) <= 1:
                continue
            simplified_mentions = []
            for m in mentions:
                simplified_mentions.append((m['sentNum'], m['startIndex'], m['endIndex'], m['text']))
            corefs.append(simplified_mentions)
        return corefs

    def ssplit(self, text):
        r_dict = self._request('ssplit, tokenize', text)
        sentence = []
        sentences = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                sentence.append(token['originalText'])

            sentences.append(TreebankWordDetokenizer().detokenize(sentence))
            #sentences.append(" ".join(sentence))
            sentence = []
        return sentences
