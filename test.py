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




with open("data/test_useless/test.txt") as f:
    for line in f:
        words = jieba.cut(line.strip())  # cut comment into words
        #words = list(set(words))
        #words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words]  # handle 23333, 6666

        print((' ').join([word.encode('utf-8') for word in words]) + ' .')



file = "data/test_useless/test.txt" #name of the plain-text file
parser = PlaintextParser.from_file(file, Tokenizer("english"))
sentences = '幸运 何其 不幸.菊仙 心疼.世间 程蝶衣.娥 郎 男儿 女娇 我本.AD17430 郎 男儿 我本.娥 郎 男儿 女娇 我本.娥 郎 男儿 女娇 我本.娥 郎 男儿 女娇 我本.一个 哭 医院 输液.泪目.哭.一周 目 泪目 后半段 心疼.死 哭.泪目.一死 终有 虞姬.致敬.第二次.清明.三周 目.死.:.:.真虞姬 假 霸王.张国荣 先生 永远 怀念.程蝶衣 张国荣.张国荣.致敬 张国荣.世间 无虞 姬.娥 程蝶衣 我本 郎 世间 男儿 女娇.'
parser = PlaintextParser.from_string(sentences, Tokenizer("english"))

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

