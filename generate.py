import numpy as np
import torch
import argparse
from operator import itemgetter
from tqdm import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from settings import model_path, best_params_multi, GEN_COUNT, gen_params
from score import score_sentence_nonlin
from tools import en_word_count, repeated_words, det_at_start, eto_at_start, pron_at_start

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate(args, tokenizer, model):
    generated_sequences = []

    encoded_prompt = tokenizer.encode(args['prompt'], add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=args['length'] + len(encoded_prompt[0]),
        temperature=args['temperature'],
        top_k=args['k'],
        top_p=args['p'],
        #num_beams=12, num_beam_groups =4, diversity_penalty =0.2,
        #length_penalty 
        #repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args['num_return_sequences'],
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("ruGPT:".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            args['prompt'] + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)

    return generated_sequences
    
    



def simplify(ref, control):
  generated_lines=[]
  best=ref
  
  current_params = best_params_multi.copy()
  # control = 0 -- maximum short, simple
  current_params['cosine'] += -0.5 + control * 0.1
  current_params['dd'] += 0.2 - control * 0.05
  current_params['en'] += -0.4 + control * 0.1
  current_params['lc'] += 0.5 - control * 0.2
  current_params['le'] += 0.5 - control * 0.1
  current_params['re'] += 0.3 - control * 0.05
  # control = 10 -- maximum cosine + EN
  
  args = {'prompt': "Text: " + ref + " Simplify ", 'temperature': gen_params['t'], 'p': gen_params['topp'], 'k': gen_params['topk'], 'num_return_sequences': GEN_COUNT, 'length': 45, 'stop_token': 'End'}

  a = generate(args, tokenizer, model)

  disp = True
  for s in a:
    s = s[len("Text: " + ref + " Simplify "):].strip()

    break_on=['\n', '?', '<s>', '</s>', 'Simplify','End','Text']

    for b in break_on:
      if s.find(b) > 1:
        s=s[:s.find(b)]

    if s.find('.') > 10:
      a.append(s[:s.find('.')])

    if len(s)<10:
      continue

    if en_word_count(s) > en_word_count(ref):
      continue

    generated_lines.append(s)

  best=ref
  bscore, ds, ls, les, rs, es, cs = score_sentence_nonlin(ref, ref, current_params) 
  bscore *= 0.8

  a = generated_lines 
  for s in a:
    s = s.replace("``", "").replace("''", "").replace(",",", ").replace("- "," - ").replace("  "," ").replace("  "," ").replace("`","").replace("<pad>","").strip()
    s = s[:1].upper() + s[1:]
    if s.upper() == s:
      s = s[:1].upper() + s[1:].lower()
    if len(s) == 0:
      continue
    score, ds, ls, les, rs, es, cs = score_sentence_nonlin(s, ref, current_params)

    if repeated_words(s) > 0:
      score -= 0.05
    if det_at_start(s) and not det_at_start(ref):
      score -= 0.05
    if eto_at_start(s) and not eto_at_start(ref):
      score -= 0.02
    if pron_at_start(s) and not pron_at_start(ref):
      score -= 0.05
    if score > bscore:
      bscore = score
      best = s
      best_pp = (ds, ls, les, rs, es, cs)

  return best
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-text', default=None, action='store', help='Input sentence to simplify')  
    parser.add_argument('-r', '--input-file', default=None, action='store', help='Input file with sentences to simplify (one line - one sentence)')  
    parser.add_argument('-O', '--output-file', default=None, action='store', help='Output file')
    parser.add_argument('-c', '--control', default=5, type=int, action='store', help='Control parameter (0 - shortest, 10 - most accurate)')  
    args = parser.parse_args()
    if args.input_text is not None:
        print(simplify(args.input_text, args.control))
    elif args.input_file is not None:
        result = []
        with open(args.input_file, 'r') as f:
            for line in tqdm(f):
                result.append(simplify(line, args.control))
        if args.output_file is not None:
            with open(args.output_file, 'w') as fout:
                for r in result:
                    fout.write(r + '\n')
        else:
            for r in result:
                print(r)
    else:
        parser.print_help()
                
    
#print(simplify('14 апреля 2003 году архиепископом Новосибирским и Бердским Тихоном пострижен в монашество с наречением имени Феодор в честь праведного Феодора Томского.'))