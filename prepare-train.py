import pandas as pd
from tqdm import tqdm
import difflib
from settings import model_path, best_params_multi, GEN_COUNT, gen_params
from score import score_sentence
from tools import en_word_count, repeated_words, det_at_start, eto_at_start, pron_at_start


train=[]
dev_df = pd.read_csv('dev_sents.csv')
for i,row in dev_df.iterrows():
  train.append("Text: %s Simplify %s EndText" % (row['INPUT:source'],row['OUTPUT:output']))
wiki_df = pd.read_csv('wiki_train_cleaned_translated_sd.csv')
pp = {'cosine': 1, 'dd': 1, 'en': 1, 'lc': 1, 'le': 1, 're': 1}
n = 0
for i, row in tqdm(wiki_df.iterrows(), total=wiki_df.shape[0], position=0, leave=True):
  score, ds, ls, les, rs, es, cs = score_sentence(row['target_y'], row['target_x'], pp)
  if difflib.SequenceMatcher(None, row['target_y'].split(' '), row['target_x'].split(' ')).ratio() > 0.8:
    continue
  if len(row['target_x']) >= 25 and len(row['target_x'].split(' ')) >= 4:
    if ds >= 0.5 and ls >= 0.65 and les >= 0.55 and rs >= 0.6 and es >= 0.65 and cs >= 0.75:
        train.append("Text: %s Simplify %s EndText" % (row['target_x'],row['target_y']))
        full.append((row['target_x'],row['target_y']))
        if i % 20 == 0:
          print(row['target_x'], '-->', row['target_y'], score, ds, ls, les, rs, es, cs)
        n += 1
        
f=open('train.txt','w',encoding='utf-8')  
for t in train:
  f.write('%s\n' % t)
f.close()