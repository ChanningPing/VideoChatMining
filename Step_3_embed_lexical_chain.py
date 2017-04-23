# -*- coding: utf-8 -*-
from __future__ import division
import gensim.models.word2vec as w2v
from dateutil.parser import parse
import math as math
from collections import Counter
import csv
import jieba
import re
import os
import pandas as pd
import sys

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
lexical_chain_dict={}
concept_dict={}

def read_word_embeddings():# read word embedding
    '''
    :return: the trained word vectors
    '''
    danmu2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))
    return danmu2vec

def read_danmu(file_name):# read danmu data, sorted by elapsed_time
    danmu = pd.read_csv(file_name, sep=',')
    danmu = danmu.sort_values(['elapse_time'], ascending=[1])
    danmu = danmu.reset_index(drop=True) #update index
    for index, row in danmu.iterrows():
        danmu.set_value(index, 'text', row['text'].replace('.', '').replace(' ', ''))
        #print(str(row['elapse_time']) + ',' + str(row['text']))
    #return danmu.head(n=100)
    return danmu



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


def constuct_lexical_chains(danmu,danmu2vec,max_silence,top_n, min_overlap):
    '''
    :param danmu: raw danmu data in pandas frame
    :param danmu2vec: word embeddings pre-trained
    :param max_silence: threshold to hold next comment in the same chain, in seconds
    :param top_n: top n most similar words
    :param min_overlap: minimum percentage of overlap
    :return: a dictionary, key: concept; value: list of lists of consecutive lexical chains
    '''
    count = 0
    simplified_danmu = []
    for index, row in danmu.iterrows():

        print('s-' + str(count) + '['+str(row['elapse_time']) +']'+ row['text'])
        words = jieba.cut(row['text']) # cut comment into words
        words = list(set(words))
        words = [re.sub(r'(.)\1+', r'\1\1', w) for w in words] # handle 23333, 6666
        words = [re.sub(r'(哈)\1+', r'\1\1', w) for w in words] # handle repetition
        words = [re.sub(r'(啊)\1+', r'\1\1', w) for w in words] # handle repetition
        words = [re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", w.decode("utf8")) for w in words]

        current_time = float(row['elapse_time'])
        word_list = []

        for w in words:
            if not w in stop_words and w:
                if w.isdigit() and (is_date(w) or not ('233' in w or '66' in w)): continue

                word_list.append(w) # used for simplified_danmu

                if w=='s': print('here1')
                if w in concept_dict: # if w in concept clusters
                    if w == 's': print('here2')
                    concept = concept_dict[w]
                    previous_time = danmu.iloc[lexical_chain_dict[concept][-1][-1][1]]['elapse_time']
                    if current_time - previous_time <= max_silence :

                        if  current_time >previous_time:
                            if w == 's': print('here3')
                            lexical_chain_dict[concept][-1].append((w,count))
                        if current_time == previous_time:
                            if w == 's': print('here4')
                            if lexical_chain_dict[concept][-1][-1][1] < index:
                                lexical_chain_dict[concept][-1].append((w, count))
                    else:
                        if w == 's':
                            print('here5')
                            print(concept.encode('utf-8'))
                        lexical_chain_dict[concept].append([(w,count)])
                        if w == 's': print( lexical_chain_dict[concept])

                else:
                    if w == 's': print('here6')
                    if not w in danmu2vec.wv.vocab: # if w is not in word embedding vocabulary
                        if w == 's': print('here4')
                        concept_dict[w] = w
                        lexical_chain_dict[w]=[[(w,count)]]
                    else:
                        if w == 's': print('here7')
                        overlap_dict = {}  # key: concept, value: a list of corresponding words
                        similar_words = danmu2vec.most_similar(w.decode('utf-8'), topn=top_n)
                        for word, similarity in similar_words:
                            if word == 's':
                                print('here8')
                                print(w.encode('utf-8'))
                            if word in concept_dict:
                                if concept_dict[word] in overlap_dict:
                                    if word == 's': print('here9')
                                    overlap_dict[concept_dict[word]].append(word)
                                else:
                                    if word == 's': print('here10')
                                    overlap_dict[concept_dict[word]] =[word]
                                if word == 's': print(overlap_dict)
                            else: # a new temporal list not sharing anything with existing

                                if w in overlap_dict:
                                    if word == 's': print('here11')
                                    overlap_dict[w].append(word)
                                else:
                                    if word == 's': print('here12')
                                    overlap_dict[w]=[word]
                                if word == 's': print(overlap_dict)
                        enough_overlap = 0
                        for key,value in overlap_dict.iteritems():
                            if 's' in value:
                                print(value)
                                print('value length=' + str(len(value)))
                                print('top_n=' + str(top_n))
                                print('min_overlap=' + str(min_overlap))
                            if len(value) / top_n >= min_overlap and key!=w: # if overlap enough, merge into existing concept
                                if 's' in value: print('here13')
                                enough_overlap = 1
                                for word in value:
                                    concept_dict[word]=key
                                    if word=='s' and 's' in concept_dict and concept_dict['s'] !='t':
                                        print(value)
                                        print('value length=' + str(len(value)))
                                        print('top_n=' + str(top_n))
                                        print('min_overlap=' + str(min_overlap))
                                        print('word='+word.encode('utf-8'))
                                        print('key='+key.encode('utf-8'))
                                if w in overlap_dict:
                                    for v in overlap_dict[w]:
                                        if v == 's': print('here14')
                                        if v == 's': print(w.encode('utf-8'))
                                        concept_dict[v] = key
                                concept_dict[w] = key


                                previous_time = danmu.iloc[lexical_chain_dict[key][-1][-1][1]]['elapse_time']
                                if current_time - previous_time <= max_silence :
                                    if current_time > previous_time:
                                        lexical_chain_dict[key][-1].append((w,count))
                                    if current_time == previous_time:
                                        if lexical_chain_dict[key][-1][-1][1]<index:
                                            lexical_chain_dict[key][-1].append((w, count))
                                else:
                                    lexical_chain_dict[key].append([(w,count)])
                                break

                        if enough_overlap ==0: # otherwise build a new concept

                            if w in overlap_dict:
                                for v in overlap_dict[w]:
                                    if v == 's': print('here14')
                                    if v == 's': print(w.encode('utf-8'))
                                    concept_dict[v]=w
                            concept_dict[w] = w
                            lexical_chain_dict[w] =[[(w,count)]]
                        #if 's' in concept_dict: print(concept_dict['s'])
        simplified_danmu.append([index, current_time,word_list])
        count = count + 1

    '''
    frequencies = []  # used for default frequency of words not in word2vec vocabulary
    for key, value in lexical_chain_dict.iteritems():
        if key in danmu2vec.wv.vocab:
            frequencies.append(danmu2vec.wv.vocab[key.decode('utf-8')].count)
    avg_frequency = sum(frequencies) / len(frequencies)
    for key, value in lexical_chain_dict.iteritems():
        frequency = -1
        if key in danmu2vec.wv.vocab:
            frequency = danmu2vec.wv.vocab[key.decode('utf-8')].count
            frequency = math.log(frequency)
        else:
            frequency = math.log(avg_frequency)
        print(key.encode('utf-8')+'['+str(frequency)+']')
        for v in value:
            for word, s_id in v:
                print(word.encode('utf-8') + '[' + str(s_id) + ']')


    for s_d in simplified_danmu:
        print(str(s_d[0])+','+(' ').join([w.encode('utf-8') for w in s_d[1]]))
    '''
    print('****************')
#    print(concept_dict['s'])
    return simplified_danmu

def similar_words(word,danmu2vec):
    similar_words = danmu2vec.most_similar(word.decode('utf-8'), topn=10)
    for word, similarity in similar_words:
        print(word)
        print(similarity)
        print(danmu2vec.wv.vocab[word.decode('utf-8')].count)

def align_comments(simplified_danmu,danmu2vec,scene_length):
    # first get default frequency for a word not in vocabulary
    frequencies = []  # used for default frequency of words not in word2vec vocabulary
    for key, value in lexical_chain_dict.iteritems():
        if key in danmu2vec.wv.vocab:
            frequencies.append(danmu2vec.wv.vocab[key.decode('utf-8')].count)
    avg_frequency = sum(frequencies) / len(frequencies)

    # re-align
    for index, row in enumerate(simplified_danmu): # each comment
        print(index)
        if row[2]:
            chain_score = {} # several chains, each with a score of importance
            for word in row[2]:
                if word=='s': print(word.encode('utf-8'))
                concept = concept_dict[word] # find concept of each word
                if word == 's':
                    print('found concept!' + concept.encode('utf-8'))
                chains = lexical_chain_dict[concept] # find corresponding chains of the concept
                if word == 's':
                    print('found chains!')
                if word == 's':
                    print(chains)
                found = 0
                found_chain = []
                for chain in chains: #  scan each chain
                    for w,s_id in chain: # each (word, sentence_id) pair in a chain
                        if index==s_id: # if this is the chain where the comment is at
                            found = 1
                            found_chain = chain
                            if word == 's':
                                print('found best chains!')
                                print(found_chain)
                            break
                    if found==1: break
                if found == 1:
                    score = 0
                    for w, s_id in found_chain:  # add up accumlative tf*idf
                        if w in danmu2vec.wv.vocab:
                            score = score + 1 / math.log(danmu2vec.wv.vocab[w.decode('utf-8')].count)
                        else:
                            score = score + 1 / math.log(avg_frequency)
                    chain_score[concept] = (score,found_chain)

            max_concept = ''
            max_score = 0
            for key, value in chain_score.iteritems():
                if key.encode('utf-8') == 's':
                    print(key.encode('utf-8')+'['+str(value[0])+']')
                if value[0] > max_score:
                    max_concept = key
                    max_score = value[0]
                if key.encode('utf-8') == 's':
                    print(chain_score)
                if key.encode('utf-8') == 's':
                    print(max_concept.encode('utf-8'))
            found_chain = chain_score[max_concept][1] # retrieve the max found_chain
            max_s_id = found_chain[0][1] # retrieve the head(0) sentence_id(1) of the max found_chain
            start_time = danmu.iloc[max_s_id]['elapse_time']
            row[1] = start_time # modifiy time of the comment

    simplified_danmu.sort(key=lambda x: x[1])
    #for s_d in simplified_danmu:
        #print('[s-'+str(s_d[0]) + ']'+str(s_d[1])+','+(' ').join([w.encode('utf-8') for w in s_d[2]]))




    return simplified_danmu
    # TODO: calculate emotion score
    # TODO: calculate topic coherence score
    # TODO: calculate intensity

def segment_danmu_to_scenes(scene_length,simplified_danmu):
    scenes = []
    current_time = scene_length
    scene =[]
    for row in simplified_danmu:
        if row[1] <= current_time:
            scene.append(row)
        else:
            scenes.append(scene)
            scene = []
            current_time = current_time + scene_length
    #for scene in scenes:
        #print('***************')
        #for row in scene:
            #print('[s-' + str(row[0]) + ']' + str(row[1]) + ',' + (' ').join([w.encode('utf-8') for w in row[2]]))
    return scenes

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

def calculate_emotion_scores(scenes,danmu2vec,emotion_lexicon ):

    for scene in scenes:
        emotion_score = {'happy': 1, 'surprise': 1, 'fear': 1, 'sad': 1, 'anger': 1}
        print('***************')
        for row in scene:
            print('[s-' + str(row[0]) + ']' + str(row[1]) + ',' + (' ').join([w.encode('utf-8') for w in row[2]]))

            for w in row[2]:
                sentence_emotion_score = {'happy': 0, 'surprise': 0, 'fear': 0, 'sad': 0, 'anger': 0}
                if w in emotion_lexicon['happy']:
                    sentence_emotion_score['happy'] = 1
                elif w in emotion_lexicon['surprise']:
                    sentence_emotion_score['surprise'] = 1
                elif w in emotion_lexicon['fear']:
                    sentence_emotion_score['fear'] = 1
                elif w in emotion_lexicon['sad']:
                    sentence_emotion_score['sad'] = 1
                elif w in emotion_lexicon['anger']:
                    sentence_emotion_score['anger'] = 1
                emotion_score = Counter(emotion_score) + Counter(sentence_emotion_score)
        print(emotion_score)
        sum = emotion_score['happy'] + emotion_score['surprise'] + emotion_score['fear']+ emotion_score['sad']+emotion_score['anger']
        entropy = 0
        max_value = 0
        for key, value in emotion_score.iteritems():
            if value > 0:
                p = value / sum
                entropy = entropy - p * math.log(p)
            if value > max_value:
                max_value = value
        score = math.log(max_value) / entropy
        print('[entropy]=' + str(entropy) + '[score]=' + str(score))















if __name__ == "__main__":
    danmu2vec = read_word_embeddings() # read word embedding
    emotion_lexicon = read_emotion_lexicon() # read emotion lexicon
    danmu = read_danmu('data/kill_bill_new.csv') # read danmu
    simplified_danmu = constuct_lexical_chains(danmu, danmu2vec, 4, 5, 0.5) # construct lexical chain
    simplified_danmu = align_comments(simplified_danmu, danmu2vec, 15) # align danmu based on lexical chain
    scenes = segment_danmu_to_scenes(10, simplified_danmu) # segment re-aligned danmu into scenes
    calculate_emotion_scores(scenes, danmu2vec, emotion_lexicon)


    #similar_words('s',danmu2vec)


