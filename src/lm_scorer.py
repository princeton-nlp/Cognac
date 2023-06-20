"""
Calculate PPL of the generated text.
python -m ci.lm_scorer
"""
import json
import yaml
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#from ci.utils import (
#    to_namespace,
#    get_data,
#    get_lm,
#    get_tokenizer,
#    get_node_children_in_text,
#    EndOfFunctionCriteria,
#)

#CONFIG = dict(
#    model_name='gpt2-xl',
#    model_name='EleutherAI/gpt-j-6B',
#    run_dir='./runs/gen_with_guidance',
#)
#args = to_namespace(CONFIG)
#print(args)

model_to_device_map = {
    'gpt2-xl': {
        0: list(range(0, 12)),
        1: list(range(12, 24)),
        2: list(range(24, 36)),
        3: list(range(36, 48)),
    },
    'EleutherAI/gpt-j-6B': {
        0: list(range(0, 7)),
        1: list(range(7, 14)),
        2: list(range(14, 21)),
        3: list(range(21, 28)),
    },
}


class LMScorer:
    def __init__(self, model_name='gpt2-xl', model=None, tokenizer=None):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            print('[LMScorer]: model reused.')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            #self.model.parallelize(model_to_device_map[model_name])
            self.model.cuda()
            #self.model.eval()
            print('[LMScorer]: new model loaded.')

    @torch.no_grad()
    def sentence_score(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()
        outputs = self.model(input_ids, labels=input_ids)
        ppl = torch.exp(outputs.loss).item()
        return ppl


def compute_ppl(path, lm_scorer):
    path = Path(path)

    predictions = []
    with open(path) as f:
        for line in f:
            prediction = json.loads(line.strip())
            predictions.append(prediction)

    ppls = []
    for prediction in predictions:
        prediction_id = prediction['datapoint']['id']
        generated_text = prediction.get('generated_text', None)

        if generated_text is None:
            generated_text = prediction['prediction']
        #print(prediction_id, generated_text)
        generated_text = [sent for sent in generated_text.split('\n') if sent]
        if len(generated_text) > 0:
            generated_text = generated_text[0]
        else:
            generated_text = ''

        if generated_text == '':
            print('Empty text.')
            continue
    
        input_ids = lm_scorer.tokenizer(generated_text, return_tensors="pt").input_ids.cuda()
        #print(input_ids.size())
        outputs = lm_scorer.model(input_ids, labels=input_ids)
        ppl = torch.exp(outputs.loss)
        if torch.isnan(ppl) or ppl > 500:
            print('NaN loss.')
            continue
        ppls.append((prediction_id, ppl.item()))
    print(f'avg: {sum(ppl for _, ppl in ppls) / len(ppls)} (n={len(ppls)})')

    save_path = path.parent / 'ppl.txt'
    with open(save_path, 'w') as f:
        for prediction_id, ppl in ppls:
            f.write(f'{prediction_id},{ppl}\n')
        f.write(f'avg,{sum(ppl for _, ppl in ppls) / len(ppls)}\n')
    print('Saved at:', save_path)
    print('----------------------------------')


@torch.no_grad()
def try_compute_ppl(lm_scorer):
    texts = [
        'There is a dog hiding behind the tree.',
        'There is a dog is a that hiding behind the tree.',
        'The dog is hiding behind the tree.',
        'The dog is hiding in cat of the movie tree.',
        'this is a dog this is a dog this is a dog this is a dog',
        'dog dog dog dog dog dog dog dog dog',
    ]
    for text in texts:
        print('text:')
        print(text)
        input_ids = lm_scorer.tokenizer(text, return_tensors="pt").input_ids.cuda()
        print([lm_scorer.tokenizer.decode(tok) for tok in input_ids[0]])
        outputs = lm_scorer.model(input_ids, labels=input_ids)
        print(outputs.loss)

        ppl = torch.exp(outputs.loss).item()
        print(f'PPL: {ppl:.4f}')
        ppl = lm_scorer.sentence_score(text)
        print(f'PPL 2: {ppl:.4f}')
        print('---')


if __name__ == '__main__':
    """
    python -m ci.lm_scorer
    """
    model_name = 'gpt2-xl'    
    #model_name = 'EleutherAI/gpt-j-6B'

    paths = [
        # wordnet
        #'./runs/wd/wordnet/wordnet_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=5.0_in=discrete_ex=discrete_trie_i2g_evalv=test/seed_0/predictions.jsonl',
        #'./ci/baselines/runs/gpt_engine="davinci"_temp=0.9_top_p=0.95_eval=-1_split=test/wordnet/predictions.jsonl',
        #'./ci/baselines/runs/gpt_engine="text-davinci-002"_temp=0.9_top_p=0.95_eval=-1_split=test/wordnet/predictions.jsonl',
        #'./runs/wd/wordnet/wordnet_nl+ex+in_gen=gpt2-xl_guide=none_a=100.0_b=5.0_in=none_ex=none_evalv=dev/seed_0/predictions.jsonl',
        #'./ci/baselines/runs/gpt_engine="davinci"_temp=0.9_top_p=0.95_eval=-1/wordnet/predictions.jsonl',
        #'./ci/baselines/runs/gpt3-legacy/wordnet/predictions.jsonl',
        #'./ci/baselines/runs/gpt3-extra-prompt/wordnet/predictions.jsonl',
        #'./runs/wd/wordnet/wordnet_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=5.0_in=discrete_ex=discrete_trie_evalv=dev/seed_0/predictions.jsonl',
        #'./runs/wd/wordnet/selfdebias_a=0.5_b=0.5_pmt=expof/seed_0/predictions.jsonl',
        #'./ci/baselines/runs/ctrl/wordnet/debug/ctrl_eval/seed_0/predictions.jsonl',

        # wikidata
        #'./runs/wd/wikidata/wikidata_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=10.0_in=discrete_ex=discrete_i2g_evalv=test/seed_0/predictions.jsonl',
        #'./ci/baselines/runs/gpt_engine="davinci"_temp=0.9_top_p=0.95_eval=-1_split=test/wikidata/predictions.jsonl',
        #'./ci/baselines/runs/gpt_engine="text-davinci-002"_temp=0.9_top_p=0.95_eval=-1_split=test/wikidata/predictions.jsonl',
        #'./runs/wd/wikidata/wikidata_nl+ex+in_gen=gpt2-xl_guide=none_a=100.0_b=10.0_in=none_ex=none_evalv=dev/seed_0/predictions.jsonl',
        #'./ci/baselines/runs/gpt_engine="davinci"_temp=0.9_top_p=0.95_eval=-1/wikidata/predictions.jsonl',
        #'./ci/baselines/runs/gpt3-legacy/wikidata/predictions.jsonl',
        #'./ci/baselines/runs/gpt3/wikidata/predictions.jsonl',
        #'./runs/wd/wikidata/wikidata_wd+ex+in_gen=gpt2-xl_guide=oracle_a=100.0_b=10.0_in=none_ex=oracle_evalv=-2/seed_0/predictions.jsonl',
        #'./runs/wd/wikidata/_right_results_bad_pred_files/wikidata_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=5.0_in=discrete_ex=binary/seed_0/predictions.jsonl',
        #'./runs/wd/wikidata/_right_results_bad_pred_files/wikidata_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=5.0_in=discrete_ex=full_extopk=20/seed_0/predictions.jsonl',
        #'./runs/wd/wikidata/wikidata_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=10.0_in=discrete_ex=discrete_beam=8_topp=0.92_return=8_impprompt/seed_0/predictions.jsonl',
        #'./runs/wd/wikidata/wikidata_wd+ex+in_gen=gpt2-xl_guide=gpt2-xl_a=100.0_b=10.0_in=discrete_ex=discrete_i2g_evalv=dev/seed_0/predictions.jsonl',
        #'./runs/wd/wikidata/selfdebias_a=0.5_b=0.5_pmt=expof/predictions.jsonl',
        './ci/baselines/runs/ctrl/wikidata/debug/ctrl_eval/seed_0/predictions.jsonl',
    ]

    lm_scorer = LMScorer(model_name=model_name)
    #try_compute_ppl(lm_scorer)
    for path in paths:
        print(path)
        compute_ppl(path, lm_scorer)