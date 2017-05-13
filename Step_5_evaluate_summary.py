# -*- coding: utf-8 -*-
from __future__ import division
import os
import csv
import sys
import re
import pickle
import gensim.models.word2vec as w2v
from dateutil.parser import parse
import operator

from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
from sumy.summarizers.lsa import LsaSummarizer as LsaSummarizer
from sumy.summarizers.kl import KLSummarizer as KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer as SumBasicSummarizer
from sumy.summarizers.luhn import LuhnSummarizer as LuhnSummarizer

import math
reload(sys)
sys.setdefaultencoding('utf8')

stop_words = ['啊','吗','，','的','一','不','在','人','有','是','为','以','于','上','他','而','后','之','来','及',
              '了','因','下','可','到','由','这','与','也','此','但','并','个','其','已','无','小','我','们','起',
              '最','再','今','去','好','只','又','或','很','亦','某','把','那','你','乃','它','吧','被','比','别',
              '趁','当','从','到','得','打','凡','儿','尔','该','各','给','跟','和','何','还','即','几','既','看',
              '据','距','靠','啦','了','另','么','每','们','嘛','拿','哪','那','您','凭','且','却','让','仍','啥',
              '如','若','使','谁','虽','随','同','所','她','哇','嗡','往','哪','些','向','沿','哟','用','于','咱',
              '则','怎','曾','至','致','着','诸','自',
              "按", "按照", "俺", "俺们", "阿", "别", "别人", "别处", "别是", "别的", "别管", "别说", "不", "不仅", "不但", "不光", "不单", "不只",
              "不外乎", "不如", "不妨", "不尽", "不尽然", "不得", "不怕", "不惟", "不成", "不拘", "不料", "不是", "不比", "不然", "不特", "不独", "不管",
              "不至于", "不若", "不论", "不过", "不问", "比方", "比如", "比及", "比", "本身", "本着", "本地", "本人", "本", "巴巴", "巴", "并", "并且",
              "非", "彼", "彼时", "彼此", "便于", "把", "边", "鄙人", "罢了", "被", "般的", "此间", "此次", "此时", "此外", "此处", "此地", "此", "才",
              "才能", "朝", "朝着", "从", "从此", "从而", "除非", "除此之外", "除开", "除外", "除了", "除", "诚然", "诚如", "出来", "出于", "曾", "趁着",
              "趁", "处在", "乘", "冲", "等等", "等到", "等", "第", "当着", "当然", "当地", "当", "多", "多么", "多少", "对", "对于", "对待", "对方",
              "对比", "得", "得了", "打", "打从", "的", "的确", "的话", "但", "但凡", "但是", "大家", "大", "地", "待", "都", "到", "叮咚", "而言",
              "而是", "而已", "而外", "而后", "而况", "而且", "而", "尔尔", "尔后", "尔", "二来", "非独", "非特", "非徒", "非但", "否则", "反过来说",
              "反过来", "反而", "反之", "分别", "凡是", "凡", "个", "个别", "固然", "故", "故此", "故而", "果然", "果真", "各", "各个", "各位", "各种",
              "各自", "关于具体地说", "归齐", "归", "根据", "管", "赶", "跟", "过", "该", "给", "光是", "或者", "或曰", "或是", "或则", "或", "何",
              "何以", "何况", "何处", "何时", "还要", "还有", "还是", "还", "后者", "很", "换言之", "换句话说", "好", "后", "和", "即", "即令", "即使",
              "即便", "即如", "即或", "即若", "继而", "继后", "继之", "既然", "既是", "既往", "既", "尽管如此", "尽管", "尽", "就要", "就算", "就是说",
              "就是了", "就是", "就", "据", "据此", "接着", "经", "经过", "结果", "及", "及其", "及至", "加以", "加之", "例如", "介于", "几时", "几",
              "截至", "极了", "简言之", "竟而", "紧接着", "距", "较之", "较", "进而", "鉴于", "基于", "具体说来", "兼之", "借傥然", "今", "叫", "将", "可",
              "可以", "可是", "可见", "开始", "开外", "况且", "靠", "看", "来说", "来自", "来着", "来", "两者", "临", "类如", "论", "赖以", "连",
              "连同", "离", "莫若", "莫如", "莫不然", "假使", "假如", "假若", "某", "某个", "某些", "某某", "漫说", "没奈何", "每当", "每", "慢说", "冒",
              "哪个", "哪些", "哪儿", "哪天", "哪年", "哪怕", "哪样", "哪边", "哪里", "那里", "那边", "那般", "那样", "那时", "那儿", "那会儿", "那些",
              "那么样", "那么些", "那么", "那个", "那", "乃", "乃至", "乃至于", "宁肯", "宁愿", "宁可", "宁", "能", "能否", "你", "你们", "您", "拿",
              "难道说", "内", "哪", "凭借", "凭", "旁人", "譬如", "譬喻", "且", "且不说", "且说", "其", "其一", "其中", "其二", "其他", "其余", "其它",
              "其次", "前后", "前此", "前者", "起见", "起", "全部", "全体", "恰恰相反", "岂但", "却", "去", "若非", "若果", "若是", "若夫", "若", "另",
              "另一方面", "另外", "另悉", "如若", "如此", "如果", "如是", "如同", "如其", "如何", "如下", "如上所述", "如上", "如", "然则", "然后", "然而",
              "任", "任何", "任凭", "仍", "仍旧", "人家", "人们", "人", "让", "甚至于", "甚至", "甚而", "甚或", "甚么", "甚且", "什么", "什么样", "上",
              "上下", "虽说", "虽然", "虽则", "虽", "孰知", "孰料", "始而", "所", "所以", "所在", "所幸", "所有", "是", "是以", "是的", "设使", "设或",
              "设若", "谁", "谁人", "谁料", "谁知", "随着", "随时", "随后", "随", "顺着", "顺", "受到", "使得", "使", "似的", "尚且", "庶几", "庶乎",
              "时候", "省得", "说来", "首先", "倘", "倘使", "倘或", "倘然", "倘若", "同", "同时", "他", "他人", "他们们", "她们", "她", "它们", "它",
              "替代", "替", "通过", "腾", "这里", "这边", "这般", "这次", "这样", "这时", "这就是说", "这儿", "这会儿", "这些", "这么点儿", "这么样", "这么些",
              "这么", "这个", "这一来", "这", "正是", "正巧", "正如", "正值", "万一", "为", "为了", "为什么", "为何", "为止", "为此", "为着", "无论",
              "无宁", "无", "我们", "我", "往", "望", "惟其", "唯有", "下", "向着", "向使", "向", "先不先", "相对而言", "许多", "像", "小", "些", "一",
              "一些", "一何", "一切", "一则", "一方面", "一旦", "一来", "一样", "一般", "一转眼", "由此可见", "由此", "由是", "由于", "由", "用来", "因而",
              "因着", "因此", "因了", "因为", "因", "要是", "要么", "要不然", "要不是", "要不", "要", "与", "与其", "与其说", "与否", "与此同时", "以",
              "以上", "以为", "以便", "以免", "以及", "以故", "以期", "以来", "以至", "以至于", "以致", "己", "已", "已矣", "有", "有些", "有关", "有及",
              "有时", "有的", "沿", "沿着", "于", "于是", "于是乎", "云云", "云尔", "依照", "依据", "依", "余外", "也罢", "也好", "也", "又及", "又",
              "抑或", "犹自", "犹且", "用", "越是", "只当", "只怕", "只是", "只有", "只消", "只要", "只限", "再", "再其次", "再则", "再有", "再者",
              "再者说", "再说", "自身", "自打", "自己", "自家", "自后", "自各儿", "自从", "自个儿", "自", "怎样", "怎奈", "怎么样", "怎么办", "怎么", "怎",
              "至若", "至今", "至于", "至", "纵然", "纵使", "纵令", "纵", "之", "之一", "之所以", "之类", "着呢", "着", "眨眼", "总而言之", "总的说来",
              "总的来说", "总的来看", "总之", "在于", "在下", "在", "诸", "诸位", "诸如", "咱们", "咱", "作为", "只", "最", "照着", "照", "直到",
              "综上所述", "贼死", "逐步", "遵照", "遵循", "针对", "致", "者", "则甚", "则", "咳", "哇", "哈",  "哉", "哎", "哗",
              "哟", "哦", "哩", "矣哉", "矣乎", "矣", "焉", "毋宁", "欤",  "嘻", "嘛", "嘘", "嘎登", "嘎", "嗳", "嗯", "嗬", "嗡嗡",
              "嗡", "喽", "喔唷", "喏", "喂", "啷当", "啪达", "啦", "啥", "啐", "啊", "唉", "哼唷", "哼", "咧", "咦", "咚", "咋", "呼哧", "呸",
              "呵", "呢", "呜呼", "呜", "呗", "呕", "呃", "呀", "吱", "吧哒", "吧", "吗", "吓", "兮", "儿", "亦", "了", "乎"]

def read_emotion_lexicon():
    emotion_lexicon = {'happy': [], 'surprise': [], 'fear': [], 'sad': [], 'anger': []}
    with open("data/manual corrected expanded 300 each.txt", "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print(line[0])
            if line[1]=='1':
                emotion_lexicon['anger'].append(line[0])
            elif line[1]=='2':
                emotion_lexicon['surprise'].append(line[0])
            elif line[1]=='3':
                emotion_lexicon['fear'].append(line[0])
            elif line[1]=='4':
                emotion_lexicon['sad'].append(line[0])
            elif line[1]=='5':
                emotion_lexicon['happy'].append(line[0])
    #print(emotion_lexicon)
    return emotion_lexicon


def is_date(string): # test if a string is a date
    if len(string)>8: return False
    if len(string)<=8 and len(string)>=4:
        try:
            parse(string[0:4])
            return True and int(string[0:4])<2018
        except ValueError:
            return False
    else:
        return False


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



def read_word_embeddings():# read word embedding
    '''
    :return: the trained word vectors
    '''
    danmu2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))

    return danmu2vec

def read_scene_data(scene_dir,emotion_lexicon,danmu2vec):
    for filename in os.listdir(scene_dir):
        print('-----------------------'+filename) # print movie name
        with open(os.path.join(scene_dir, filename)) as csvfile:  # read a scene
            scene_sentences = list(csv.reader(csvfile))
            #generate_our_summary(filename, scene_sentences, emotion_lexicon, danmu2vec)
            generate_embedding_lexical_chain_summary(filename, scene_sentences, emotion_lexicon)
            generate_benchmark_summary(filename)

def generate_evaluation(file_name, emotion_lexicon):
    scene_summary_dict = {}
    scene_dir = 'data/text_summary/'
    with open(file_name, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if line[0] in scene_summary_dict:
                scene_summary_dict[line[0]].append(line[1])
            else:

                scene_summary_dict[line[0]] = [line[1]]
    for movie_scene, sentences in scene_summary_dict.iteritems():
        print('=================' + movie_scene + '===================')
        with open(os.path.join(scene_dir, movie_scene + '.txt')) as csvfile:  # read a scene
            scene_sentences = list(csv.reader(csvfile))
            #generate_our_summary(filename, scene_sentences, emotion_lexicon, danmu2vec)
            num_summary = len(sentences)
            generate_embedding_lexical_chain_summary(movie_scene, scene_sentences, emotion_lexicon,num_summary)
            generate_benchmark_summary(movie_scene,num_summary)





def generate_our_summary(filename,scene_sentences,emotion_lexicon,danmu2vec):
    concept_vector = {}
    concept_sentence = {}
    concept_dict = load_obj(filename.split('_')[0] + '_concept_dict')
    for index, s in enumerate(scene_sentences):
        # pre-processing
        words = s[0].split()
        words = list(set(words))
        words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words]  # handle 23333, 6666
        words = [re.sub(r'(哈)\1+', r'\1\1', w) for w in words]  # handle repetition
        words = [re.sub(r'(啊)\1+', r'\1\1', w) for w in words]  # handle repetition
        words = [re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）:：；《）《》“”()»〔〕-]+", "", w.decode("utf8")) for w
                 in words]
        words = [w for w in words if not w in stop_words]
        for w in sorted(words):  # for each word
            # print(w.encode('utf-8'))
            if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)):
                break
            if w:  # save scene info into concept_vector and concept_scene
                if w in emotion_lexicon['happy']:
                    concept = '哈哈'
                elif w in emotion_lexicon['surprise']:
                    concept = '卧槽'
                elif w in emotion_lexicon['fear']:
                    concept = '可怕'
                elif w in emotion_lexicon['sad']:
                    concept = '泪目'
                elif w in emotion_lexicon['anger']:
                    concept = '气死了'
                #elif w in concept_dict:
                    #concept = concept_dict[w]
                else:
                    concept = w

                if concept in concept_vector:
                    concept_vector[concept].append(index)  # record the sentence id where each concept occurs
                else:
                    concept_vector[concept] = [index]
                # another dict used for accessing concept by sentence_id
                if index in concept_sentence:  # row[0] is sentence id
                    concept_sentence[index].append(concept)
                else:
                    concept_sentence[index] = [concept]

    #print the concept chains
    for key, value in concept_vector.iteritems():
        print('[' + key.encode('utf-8') + ']' + (' ').join([str(s_id) for s_id in value]))
    # calculate concept importance
    concept_importances = {}
    for key, value in concept_vector.iteritems():
        # concept_importance = len(list(set(value))) / concept_idf
        concept_importance = len(list(set(value)))
        concept_importances[key] = concept_importance
    sorted_concept_importances = sorted(concept_importances.items(), key=operator.itemgetter(1), reverse=True)

    print('===========Our method============')

    # use previous sentence set subtract later sentence set of concept
    valid_concepts = []
    intersection = []
    for index, concept_importance in enumerate(sorted_concept_importances):
        original_vector = list(concept_vector[concept_importance[0]])
        concept_vector[concept_importance[0]] = list(
            set(concept_vector[concept_importance[0]]) - set(intersection))
        intersection = list(set(intersection) | set(original_vector))
        if concept_vector[concept_importance[0]]:
            valid_concepts.append(concept_importance[0])

    # print(concept_sentence)
    for index, concept in enumerate(valid_concepts):
        s_ids = list(set(concept_vector[concept]))  # all sentences of this concept
        sentence_scores = {}
        for s_id in s_ids:  # for each sentence
            score = 0
            # print('[s_id]='+str(s_id))

            for c in list(set(concept_sentence[s_id])):
                # print(concept_sentence[s_id])
                if c in danmu2vec.wv.vocab:
                    concept_idf = math.log(danmu2vec.wv.vocab[c.decode('utf-8')].count)
                else:
                    concept_idf = math.log(600)
                score = score + len(list(set(concept_vector[c]))) * concept_idf
                #score = score + len(list(set(concept_vector[c])))

            # score is the total score excluding the key concept, and use 1 as a low estimate
            sentence_scores[s_id] = score / len(list(set(concept_sentence[s_id])))
            #sentence_scores[s_id] = score
        sentence_scores = sorted(sentence_scores.items(), key=operator.itemgetter(1), reverse=True)
        best_s_id = sentence_scores[0][0]
        print('[' + concept.encode('utf-8') + '][' + str(best_s_id) + ']' + '][' + str(sentence_scores[0][1]) + ']' +
              scene_sentences[best_s_id][0].replace(' ', '').encode('utf-8'))
        # if index / len(valid_concepts) >= compression_rate:
        if index == 9:
            break
def word_2_concept(w,emotion_lexicon,concept_dict):
    if w in emotion_lexicon['happy']:
        concept = '哈哈'
    elif w in emotion_lexicon['surprise']:
        concept = '卧槽'
    elif w in emotion_lexicon['fear']:
        concept = '可怕'
    elif w in emotion_lexicon['sad']:
        concept = '泪目'
    elif w in emotion_lexicon['anger']:
        concept = '气死了'
    #elif w in concept_dict:
        #concept = concept_dict[w]
    else:
        concept = w
    return concept

def purity_words(words):
    words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words]  # handle 23333, 6666
    words = [re.sub(r'(哈)\1+', r'\1\1', w) for w in words]  # handle repetition
    words = [re.sub(r'(啊)\1+', r'\1\1', w) for w in words]  # handle repetition
    words = [re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，.。？?、~@#￥%……&*（）:：；《）《》“”()»〔〕-]+", "", w.decode("utf8")) for w in
             words]
    words = [w for w in words if not w in stop_words]
    return words

def generate_embedding_lexical_chain_summary(filename,scene_sentences,emotion_lexicon,num_summary):
    concept_dict = load_obj(filename.split('_')[0] + '_concept_dict')
    print('===========Our method============')
    # get the overal number of tokens in the scene
    word_length = 0
    for index, s in enumerate(scene_sentences):
        # pre-processing
        s[0] = s[0].replace('.','')
        words = s[0].split()
        words = purity_words(words)
        word_length += len(words)
    # count concept (word) occurrence
    word_dict = {}
    for index, s in enumerate(scene_sentences):
        words = s[0].split()
        words = purity_words(words)
        for w in sorted(words):  # for each word
            # print(w.encode('utf-8'))
            if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)):
                break
            if w:  # save scene info into concept_vector and concept_scene
                concept = word_2_concept(w, emotion_lexicon,concept_dict)

                if concept in word_dict:
                    word_dict[concept.decode('utf-8')] += 1
                else:
                    word_dict[concept.decode('utf-8')] = 1
    # scale word occurrence to [0,1]
    for key, value in word_dict.iteritems():
        word_dict[key] = value / word_length

    #for key, value in word_dict.iteritems():
        #print(key.encode('utf-8') + ':' + str(value))
    count = 0
    #num_summary = 8
    while count < num_summary:
        # calcualte sentence score
        sentence_scores = []
        for index, s in enumerate(scene_sentences):
            sentence_probabilities = []
            words = s[0].split()
            words = purity_words(words)
            for w in sorted(words):  # for each word
                # print(w.encode('utf-8'))
                if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)):
                    break
                if w:  # save scene info into concept_vector and concept_scene
                    concept = word_2_concept(w, emotion_lexicon,concept_dict)
                    sentence_probabilities.append(word_dict[concept.decode('utf-8')])
                    #print(concept.encode('utf-8') + str(word_dict[concept]))
            if len(sentence_probabilities):
                sentence_scores.append(sum(sentence_probabilities) / len(sentence_probabilities))
            else:
                sentence_scores.append(0)
            #print(sentence_probabilities)

        max_s_id, max_score = max(enumerate(sentence_scores), key=operator.itemgetter(1))
        print('[' + str(max_score) + ']' + scene_sentences[max_s_id][0].encode('utf-8'))
        words = scene_sentences[max_s_id][0].split()
        words = purity_words(words)
        for w in sorted(words):  # for each word
            # print(w.encode('utf-8'))
            if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)):
                break
            if w:  # save scene info into concept_vector and concept_scene
                concept = word_2_concept(w, emotion_lexicon,concept_dict)
                word_dict[concept.decode('utf-8')] = -0.5
                #print(concept.encode('utf-8') + str( word_dict[concept.decode('utf-8')]))

        #for key,value in word_dict.iteritems():
            #print(key.encode('utf-8') + ':'+ str(value))
        scene_sentences.pop(max_s_id)
        count += 1



def calculate_ROGUE_1():
    A = 1
    # TODO: read golden standard data

def calcualte_ROGUE_2():
    A = 1





def generate_benchmark_summary(filename,num_summary):

    parser = PlaintextParser.from_file('data/text_summary/' + filename + '.txt', Tokenizer("english"))
    print('=========== Basic Sum ============')
    summarizer = SumBasicSummarizer()
    summary = summarizer(parser.document, num_summary)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('=========== LSA ============')
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_summary)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('===========LexRank============')
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_summary)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('===========KL Divergence============')
    summarizer = KLSummarizer()
    summary = summarizer(parser.document, num_summary)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence

    print('===========Luhn============')
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, num_summary)  # Summarize the document with 5 sentences
    for sentence in summary:
        print sentence
if __name__ == "__main__":
    scene_dir = 'data/text_summary/'
    danmu2vec = read_word_embeddings()
    emotion_lexicon = read_emotion_lexicon()
    #read_scene_data(scene_dir,emotion_lexicon,danmu2vec)
    file_name = 'data/summary_golden_standard/summary.csv'
    generate_evaluation(file_name, emotion_lexicon)