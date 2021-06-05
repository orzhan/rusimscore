from tools import cosine, dep_depth_score, lexical_complexity_score, length_score, reading_ease_score, entity_score


def nonlin(val):
  if val < 0.55:
    val = (val - 0.5) / 2 + 0.5
  elif val < 0.6:
    val = (val - 0.5) / 1.9 + 0.5
  elif val < 0.65:
    val = (val - 0.5) / 1.8 + 0.5
  elif val < 0.7:
    val = (val - 0.5) / 1.7 + 0.5
  elif val < 0.8:
    val = (val - 0.5) / 1.4 + 0.5
  elif val < 0.9:
    val = (val - 0.5) / 1.1 + 0.5
  else:
    val = val
  return val

def score_sentence(s, s0, params):
  dd, lc, le, re, en, cw = params['dd'], params['lc'], params['le'], params['re'], params['en'], params['cosine']
  cs = ((cosine(s, s0)/2+0.5) - 0.5)*2
  ds = (dep_depth_score(s))
  ls = (lexical_complexity_score(s))
  les = (length_score(s, s0))
  rs = (reading_ease_score(s))
  es = (entity_score(s, s0))
  ret = (cs ** cw) * (ds ** dd) * (ls ** lc) * (les ** le) * (rs ** re) * (es ** en)
  return ret, ds, ls, les, rs, es, cs
  
def score_sentence_nonlin(s, s0, params):
  dd, lc, le, re, en, cw = params['dd'], params['lc'], params['le'], params['re'], params['en'], params['cosine']
  #print(dd, lc, le, re, en)
  #print(s, s0)
  cs = (nonlin(cosine(s, s0)/2+0.5) - 0.5)*2
  ds = nonlin(dep_depth_score(s))
  ls = nonlin(lexical_complexity_score(s))
  les = nonlin(length_score(s, s0))
  rs = nonlin(reading_ease_score(s))
  es = nonlin(entity_score(s, s0))
  ret = (cs ** cw) * (ds ** dd) * (ls ** lc) * (les ** le) * (rs ** re) * (es ** en)
  return ret, ds, ls, les, rs, es, cs