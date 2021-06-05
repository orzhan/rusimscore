from laserembeddings import Laser
from sklearn.metrics.pairwise import cosine_similarity
from itertools import islice

import math
import pandas as pd 

laser = Laser()

ru_words = None
try:
    df = pd.read_csv('ru_full.txt', header=None, sep=' ')
    df.columns=['word','count']
    total_count = df['count'].sum()
    #ru_words = {}
    #for i,row in df.iterrows():
    #  ru_words[row['word']] = row['count'] / total_count
    ru_words = dict(zip(df.word, df['count']))
    #ru_words = df.to_dict('list')
    top_words = [x[0] for x in list(islice(ru_words.items(),1000))]

    del df
except FileNotFoundError as ex:
    print('Please run download.sh to download ru_full.txt')
    print(ex)



def cosine(s,ref):
  
  embeddings = laser.embed_sentences([s,ref], lang='ru')
  return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]



from natasha import  MorphVocab, Doc,   NewsEmbedding,  NewsMorphTagger, Segmenter, NewsNERTagger,  NewsSyntaxParser
morph_vocab = MorphVocab()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)
syntax_parser = NewsSyntaxParser(emb)



tag_text_cache = {}
def tag_text(text):
  if not (text in tag_text_cache):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)
    doc.parse_syntax(syntax_parser)
    for span in doc.spans:
      span.normalize(morph_vocab)
    tag_text_cache[text] = doc

  return tag_text_cache[text]


# 2. Dep tree depth
def dep_tree_depth(text):
  doc = tag_text(text)
  dmax = 0
  for s in doc.sents:
    seen = []
    for ti in s.tokens:
      look = [(ti.id, 0)]
      while len(look) > 0:
        z, zd = look[0]
        seen.append(z)
        #print('look take', z, zd)
        look = look[1:]
        for token in s.tokens:
          if token.head_id == z and token.head_id != token.id:
            if not token.id in seen:
              look.append((token.id, zd + 1))
            #print('look add', token.id, zd + 1)
            dmax = max(dmax, zd + 1)
  return dmax

def dep_depth_score(s):
  n = dep_tree_depth(s)
  if n <= 2:
    return 1.0
  elif n == 3:
    return 0.9
  elif n == 4:
    return 0.7
  else:
    return 0.5

def remove_text_in_brackets(test_str):
    ret = ''
    skip2c = 0
    for i in test_str:
        if i == '(':
            skip2c += 1
        elif i == ')'and skip2c > 0:
            skip2c -= 1
        elif skip2c == 0:
            ret += i
    return ret

def match_entities(w, w0):
  if len(w.split(' ')) == 1 and len(w0.split(' ')) == 1:
    return w.lower() == w0.lower()
  s = set(w.lower().split(' '))
  s0 = set(w0.lower().split(' '))
  if len(s.intersection(s0)) >= 1:
    return True
  else:
    return False


def count_matching_entities(e, e0):
  #return len(list(e.intersection(e0)))
  n = 0
  for w in e:
    found = False
    for w0 in e0:
      if match_entities(w, w0):
        found = True
        break
    if found:
      n += 1
  return n


# 3. Keep Entities
entities_cache = {}

def get_entities(text):
  text = remove_text_in_brackets(text)
  if not (text in entities_cache):
    doc = tag_text(text)
    #for token in doc.tokens:
    #  token.lemmatize(morph_vocab)

  # print(doc.spans)

    ents = [s.normal for s in doc.spans]

    ents=set(ents)

    entities_cache[text] = ents

  return entities_cache[text]
  #return set([text[s.start:s.stop] for s in ner(text).spans])


def entity_score(s, s0):
  e = get_entities(s)
  e0 = get_entities(s0)
  #print(e)
  #print(e0)
  k = count_matching_entities(e, e0)
  n = len(e0.union(e))
  #print('match',k,'union',n)
  k = min(4, k)
  n = min(4, n)
  #print('n',n,'k',k)
  if max(k,n) == 0:
    return 1
  #print(k,n,e0,e)
  return 1 -  0.5 * (abs(k-n)/max(k,n))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# entity_score('Александр Невский был во Франции', 'Путешествие Александра Невского во Францию')


# 4. Lexical complexity

#ru_words[ru_words.word == 'порох'].iloc[0]['count']  / total_count


#ru_words

def word_frequency(s):
  if not (s in ru_words):
    return 0
  return ru_words[s]  / total_count

def lexical_complexity(text):
  doc = tag_text(text)
  ents = list(get_entities(text))
  for s in doc.sents:
    f = 0
    n = 0
    worst = 1.0
    for token in s.tokens:
      if token.text in ents:
        continue
      if token.pos == 'PUNCT' or token.pos == 'NUM' or token.pos == 'ADP' or token.pos == 'PROPN':
        continue
      #print(token)
      v = word_frequency(token.text)
      #print(token.text, v)
      if v == 0:
        #v = 1e-16
        continue
      if v > 0:
        f += math.log(v)
        n += 1
      worst = min(worst, f)
    if n > 0:
      f = f / n
  return f, worst



def lexical_complexity_score(s):
  score, worst = lexical_complexity(s)
 # score0 = lexical_complexity(s0)
  return 1.0 + (min(max(score, -24), -4) + 4) / 40 +  0.0025 + (min(max(worst, -200), 0) + 0) / 1600







# 5. Sentence length
import re

def sentence_length(s):
  doc = tag_text(s)
  return len([x for x in doc.tokens if (re.match(r'[А-Яа-я0-9]', x.text) is not None)  ])

def length_score(s, s0):
  n = sentence_length(s)
  n0 = sentence_length(s0)
  if n > n0:
    return 0.5
  elif n0 < 6:
    return 1
  elif n >= 6:
    return 1 - (n / n0) * 0.5
  else:
    return n / 6 

# 6. Ease of reading
def reading_ease(s):
  word_count = sentence_length(s)
  syllable_count = len(re.findall(r'[аяоёуюиыеэАЯОЁУЮИЫЕЭ]', s))
  avg_words_p_sentence = word_count / len(tag_text(s).sents)

  score = 0.0
  analyzedVars = {
            'word_cnt': float(word_count),
            'syllable_cnt': float(syllable_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
  } 
  #print(analyzedVars)
  if analyzedVars['word_cnt'] > 0.0:
    score = 206.835 - 1.52 * analyzedVars['avg_words_p_sentence'] - 65.14 * (analyzedVars['syllable_cnt']/ analyzedVars['word_cnt'])
  return round(score, 4)

def reading_ease_score(s):
  v = reading_ease(s)
  v = max(-100, min(100, v))/400
  return 0.75 + v
  



import re

en = re.compile('[A-Za-z]')

def en_word_count(s):
  words = s.split(' ')
  return sum([1 if re.match(en, w) else 0 for w in words ])


def repeated_words(s):
  doc = tag_text(s)
  found_repeat = 0
  for sent in doc.sents:
    words = []
    for token in sent.tokens:
      if len(token.text) > 3:
        if not (token.pos == 'PUNCT' or token.pos == 'NUM' or token.pos == 'ADP' or token.pos == 'PROPN'):
          words.append(token.text.lower())
    if len(list(set(words))) < len(words):
      found_repeat += 1
  return found_repeat


def det_at_start(s):
  doc = tag_text(s)
  for token in doc.tokens[:2]:
    if token.pos == 'DET':
      return True
  return False

def pron_at_start(s):
  doc = tag_text(s)
  for token in doc.tokens[:2]:
    if token.pos == 'PRON':
      return True
  return False

def pron_count(s):
  doc = tag_text(s)
  n = 0
  for token in doc.tokens[:3]:
    if token.pos == 'PRON':
      n += 1
  return n

def unknown_words(text):
  doc = tag_text(text)
  ents = list(get_entities(text))
  words = []
  for token in doc.tokens:
    if token.text in ents:
      continue
    if token.pos == 'PUNCT' or token.pos == 'NUM' or token.pos == 'ADP' or token.pos == 'PROPN':
      continue
    #print(token)
    v = word_frequency(token.text)
    if v == 0:
      words.append(token.text)
  return set(words)

def new_unknown_word_count(s, s0):
  return len(list(unknown_words(s).difference(unknown_words(s0))))

def num_adj(text):
  doc = tag_text(text)
  return sum([1 for x in doc.tokens if x.pos == 'ADJ'])

def num_genitive(text):
  doc = tag_text(text)
  return sum([1 for x in doc.tokens if x.pos == 'NOUN' and 'Case' in x.feats and x.feats['Case'] == 'Gen'])


def adj_score(s,s0):
  n=num_adj(s)
  n0=num_adj(s0)
  if n>=n0:
    return 0.5
  if n0 == 0 or n == 0:
    return 1.0
  v = n0 / n
  # v==2 -> 1.0
  # v==1 -> 0.5
  v = min(2, max(1, v))
  return (v-1)/2 + 0.5


def gen_score(s,s0):
  n=num_genitive(s)
  n0=num_genitive(s0)
  if n>=n0:
    return 0.5
  if n0 == 0 or n == 0:
    return 1.0
  v = n0 / n
  # v==2 -> 1.0
  # v==1 -> 0.5
  v = min(2, max(1, v))
  return (v-1)/2 + 0.5


def eto_at_start(text):
  doc = tag_text(text)
  if text.find('- это') != -1:
    return False
  for token in doc.tokens[:4]:
    if token.text == 'это':
      return True
  return False



def length_absolute_score(s):
  n = sentence_length(s)
  if n <= 9:
    return 1.0
  if n > 20:
    return 0.5
  return 1 - 0.5 * ((n-9)/11)




