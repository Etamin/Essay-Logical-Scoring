from openie import StanfordOpenIE
import os
import sys
import spacy
import neuralcoref
import stanza
from nltk.parse import stanford
from nltk.parse.stanford import StanfordParser

from nltk.tree import ParentedTree, Tree
from numpy import *

import warnings

warnings.filterwarnings('ignore')
java_path = "C:/Program Files/Java/jdk-11.0.11/bin/java.exe"
os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_PARSER'] = './model/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = './model/stanford-parser-4.2.0-models.jar'

def sentence_split(str_centence):
    list_ret = list()
    for s_str in str_centence.split('.'):
        if '?' in s_str:
            list_ret.extend(s_str.split('?'))
        elif '!' in s_str:
            list_ret.extend(s_str.split('!'))
        else:
            if s_str != "":
                list_ret.append(s_str)
    return list_ret



def search(dic,start):
    queue=[]
    queue.append(start)
    bfsflag=set()
    bfsflag.add(start)
    while queue:
        v=queue.pop(0)
        bfsflag.add(v)
        for item in dic[v]:
            if (not (item in bfsflag))& (not (item in queue)) :
                queue.append(item)           
    return list(bfsflag)

def bfs(dic):
    max=0
    for key in dic:
        v=search(dic,key)
        if (len(v)>max):
            max=len(v)
    return (max)

    


if __name__ =="__main__":

    parser = StanfordParser()
    corenlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)

    # with StanfordOpenIE() as client:
    path=sys.argv[1]
    files= os.listdir(path)
    list1 = []
    rate=[]
    for file in files:
        if not os.path.isdir(file):
            f = open(path+"/"+file,'r',encoding='utf-8')
            content=f.read()
            content=content.replace('\n', '').replace('\r', '')
            doc = nlp(content)
            content_cor= doc._.coref_resolved
            documents= nlp(content_cor)
            rel={}
            words=set()
            for sentence in list(documents.sents):
                wordinsent=set()
                wordvecinset=set()
                for word in sentence:  
                    # print(word)
                    
                    if str(word.tag_) in ['NN','NNS','NNP','NNPS','WP']:
                        # print(word, word.tag_)
                        wordinsent.add(word.text)
                        # wordvecinset.add(word)
                        words.add(word)
                
                se1=list(wordinsent)
                for w in se1:
                    if not(w in rel):
                        rel[w]=wordinsent
                    elif w in rel:
                        rel[w]=rel[w]|wordinsent
            for wd1 in words:
                for wd2 in words:
                    if (wd1.text!=wd2.text) & (wd1.similarity(wd2)>=0.8):
                        # print(wd1.text+" "+wd2.text)
                        rel[wd1.text].add(wd2.text)
                        rel[wd2.text].add(wd1.text)
            rate.append((bfs(rel)/len(rel)))
    print("Avg:"+str(mean(rate)))
