# -*- coding: utf-8 -*-
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.lsa import LsaSummarizer as LsaSummarizer
from sumy.summarizers.kl import KLSummarizer as KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer as SumBasicSummarizer
from sumy.summarizers.luhn import LuhnSummarizer as LuhnSummarizer

import jieba

import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')




with open("data/test_useless/test.txt") as f:
    for line in f:
        words = jieba.cut(line.strip())  # cut comment into words
        #words = list(set(words))
        #words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words]  # handle 23333, 6666

        print((' ').join([word.encode('utf-8') for word in words]) + ' .')



file = "data/test_useless/test.txt" #name of the plain-text file
parser = PlaintextParser.from_file(file, Tokenizer("english"))
sentences = '幸何其不幸. 菊仙心疼. 世间程蝶衣.  娥 郎 男儿 女娇 我本. 郎 男儿 我本.'.decode('utf-8')
#parser = PlaintextParser.from_string(sentences, Tokenizer("english"))
print(parser.document)

print('=========== Basic Sum ============')
summarizer = SumBasicSummarizer()
summary = summarizer(parser.document, 3) #Summarize the document with 5 sentences
for sentence in summary:
    print sentence

print('=========== LSA ============')
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 3) #Summarize the document with 5 sentences
for sentence in summary:
    print sentence


print('===========LexRank============')
summarizer = LexRankSummarizer()
summary = summarizer(parser.document, 3) #Summarize the document with 5 sentences
for sentence in summary:
    print sentence

print('===========KL Divergence============')
summarizer = KLSummarizer()
summary = summarizer(parser.document, 3) #Summarize the document with 5 sentences
for sentence in summary:
    print sentence

print('===========Luhn============')
summarizer = LuhnSummarizer()
summary = summarizer(parser.document, 3) #Summarize the document with 5 sentences
for sentence in summary:
    print sentence

