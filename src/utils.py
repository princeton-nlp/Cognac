import json
import yaml
import random
from pathlib import Path
from ast import literal_eval
from argparse import Namespace
from collections import defaultdict
from os.path import dirname, abspath, join

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

random.seed(0)


def top_p_sampling(logits, top_p):
    """
    logits: (1, vocab_size)
    Code taken from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits = logits.squeeze(0)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    #print(cumulative_probs.tolist()[100])
    #print(cumulative_probs.tolist()[200])
    #print(cumulative_probs.tolist()[500])
    #print(cumulative_probs.tolist()[1000])
    #print(cumulative_probs.tolist()[2000])
    #print(cumulative_probs.tolist()[5000])
    #print(cumulative_probs.tolist()[10000])
    #print(cumulative_probs.tolist()[20000])
    #print(cumulative_probs.tolist()[-1])
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits


class EndOfFunctionCriteria(StoppingCriteria):
    """
    Code taken from: https://github.com/huggingface/transformers/blob/1d651868d64e8f54f7bf6b687fbcdac832039334/examples/research_projects/codeparrot/scripts/human_eval.py#L25
    Custom `StoppingCriteria` which checks if all generated functions in 
    the batch are completed.
    """

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """
        Returns true if all generated sequences contain any of the 
        end-of-function strings.
        """
        decoded_generations = \
            self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([
                stop_string in decoded_generation
                for stop_string in self.eof_strings
            ]))
        return all(done)


def to_namespace(config):
    return Namespace(**config)


def prepare_args(args):
    if args.model_type in ('classification', 'prompt-tune', 'fine-tune'):
        args.metric_name = 'Accuracy'
    elif args.model_type == 'regression':
        args.metric_name = 'Class=1 Precision'
    else:
        raise ValueError(f'Unknown model_type: {args.model_type}')
    return args


def update_args(args):
    """
    Full in null args for later-trained model.
    """
    if not hasattr(args, 'loss_type'):
        setattr(args, 'loss_type', 'ce')
    return args


def get_wikidata_p_text(p_name, p_value):
    if p_name == 'place_of_birth':
        fill_text = f'people who were born in {p_value}'
    elif p_name == 'place_of_death':
        fill_text = f'people who died in {p_value}'
    elif p_name == 'occupation':
        fill_text = p_value
    elif p_name == 'country_of_citizenship':
        fill_text = f'people who are citizens of {p_value}'
    elif p_name == 'academic_degree':
        fill_text = f'people who hold a degree in {p_value}'
    elif p_name == 'educated_at':
        fill_text = f'people who had their education at {p_value}'
    return fill_text


def cleanup_gen_text(text):
    """
    Cleanup items
    - drop the last unfinished sentence (should finish with `==`)

    The normal case: `len(text.split('\n')) == 3`.
    """
    sents = text.strip().split('\n')
    return sents[0]
#    print(sents)
#    if len(sents) > 2 and not sents[-1].endswith('=='):
#        sents = sents[:-1]
#    return '\n'.join(sents)


def reformat_text(text):
    text_tokens = text.split(' ')
    text = ' '.join(text_tokens)
    return text


def get_hierarchy_path_to_children(path):
    hierarchy_path_to_children = dict()
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            hierarchy_path = obj['hierarchy_path']
            children = obj['children']
            hierarchy_path_to_children[tuple(hierarchy_path)] = children
    return hierarchy_path_to_children


def get_hierarchy(path=None):
    if path is None:
        path = '/path/to/your/topic_to_leafs.json'
    with open(path) as f:
        hierarchy_ = json.load(f)
    hierarchy = dict()
    for topic, leafs in hierarchy_.items():
        new_topic = topic.replace('_', ' ')
        new_leafs = [l.replace('_', ' ') for l in leafs]
        hierarchy[new_topic] = new_leafs
    return hierarchy


class WikidataHierarchy:
    def __init__(self, q_to_p, p_to_q):
        self.q_to_p = q_to_p
        self.p_to_q = p_to_q

    def __contains__(self, key):
        if self.__getitem__(key):
            return True
        else:
            return False
    
    def __getitem__(self, p):
        """
        For a given `p` (i.e., topic or constraint), return a list of all Q's
        that have this `p`. 
        Return list like [(Q7339, 'Margot Frank'), ...].
        """
        if isinstance(p, list):
            p = tuple(p)
        
        return self.p_to_q.get(p, [])
    

def get_wikidata_hierarchy():
    from evaluation.scripts.build_wikidata_dataset import (
        load_ranked_properties,
        load_all_entities
    )
    WIKIDATA_PATH = Path('path/to/your/qid2title.json')
    q_to_p, p_to_q = load_all_entities(WIKIDATA_PATH)
    hierarchy = WikidataHierarchy(q_to_p, p_to_q)
    return hierarchy


def normalize_datapoint(datapoint, args, hierarchy):
    if args.data == 'wordnet':
        return datapoint
    elif args.data == 'wikidata':
        normalized = dict()
        if 'example_id' in datapoint:
            normalized['id'] = datapoint['example_id']
        elif 'id' in datapoint:
            normalized['id'] = datapoint['id']

        if 'text' in datapoint:
            normalized['context'] = datapoint['text']
        elif 'context' in datapoint:
            normalized['context'] = datapoint['context']

        normalized['topic'] = (
            tuple(datapoint['p'])
            if 'topic' not in datapoint
            else datapoint['topic']
        )
        normalized['gen_qs'] = datapoint['gen_qs']
        normalized['gen_text'] = datapoint['gen_text']

        if 'constraint' in datapoint:
            normalized['constraint'] = datapoint['constraint']
        else:
            constraint_candidates = defaultdict(int)
            for gen_q in datapoint['gen_qs']:
                gen_ps = hierarchy.q_to_p[tuple(gen_q)]
                for p_name, p_values in gen_ps.items():
                    for p_value in p_values:
                        constraint_candidates[(p_name, p_value)] += 1
            
            SKIP = {}
            selected_constraint = None
            for constraint in constraint_candidates.keys():
                if normalized['topic'][0] != constraint[0] and constraint[0] not in SKIP:
                    selected_constraint = constraint
                    break
            
            normalized['constraint'] = selected_constraint
        return normalized
    else:
        raise ValueError(f'Data arg {args.data} not recognized.')


def get_data(args, randomize=False, hierarchy_only=False):
    if args.data == 'wordnet':
        hierarchy = get_hierarchy(args.hierarchy_path)
    elif args.data == 'wikidata':
        # Return the modifier to update the datapoint.
        hierarchy = get_wikidata_hierarchy()
    else:
        raise ValueError(f'Data arg {args.data} not recognized.')
    
    if hierarchy_only:
        return hierarchy
    
    datasets = defaultdict(list)
    data_paths = {
        'train': args.train_path,
        'dev': args.dev_path,
        'test': args.test_path,
    }
    for dataset_split, dataset_path in data_paths.items():
        if dataset_path is not None:
            with open(dataset_path) as f:
                for line in f:
                    datapoint = json.loads(line.strip())
                    datapoint = normalize_datapoint(datapoint, args, hierarchy)

                    topic = datapoint['topic']
                    constraint = datapoint['constraint']
                    if (
                        args.data == 'wordnet' and
                        (
                            constraint not in hierarchy[topic] or 
                            constraint == topic or 
                            constraint not in hierarchy
                        )
                    ):
                        # skipping bad data
                        continue
                    
                    if args.data == 'wikidata' and constraint is None:
                        continue

                    datasets[dataset_split].append(datapoint)
        if randomize:
            random.shuffle(datasets[dataset_split])

        num_datapoints = getattr(args, dataset_split + '_num_datapoints', 100000000)
        datasets[dataset_split] = datasets[dataset_split][:num_datapoints]
    return datasets, hierarchy, None


def get_model_class(model_name):
    from ci.train_guidance_model import (
        MLPRegressor,
        MLPClassifier,
        PromptTuningModel,
        FineTuningModel,
    )
    NAME_TO_CLASS = {
        'MLPRegressor': MLPRegressor,
        'MLPClassifier': MLPClassifier,
        'PromptTuningModel': PromptTuningModel,
        'FineTuningModel': FineTuningModel,
    }
    return NAME_TO_CLASS[model_name]


def put_model_on_gpus(model, model_name, num_devices):
    if model_name == 'gpt2-xl': 
        if num_devices == 8:
            device_map = {
                0: list(range(0, 6)),
                1: list(range(6, 12)),
                2: list(range(12, 18)),
                3: list(range(18, 24)),
                4: list(range(24, 30)),
                5: list(range(30, 36)),
                6: list(range(36, 42)),
                7: list(range(42, 48)),
            }
            model.parallelize(device_map)
        elif num_devices == 4:
            device_map = {
                0: list(range(0, 12)),
                1: list(range(12, 24)),
                2: list(range(24, 36)),
                3: list(range(36, 48)),
            }
            model.parallelize(device_map)
        elif num_devices == 2:
            device_map = {
                0: list(range(0, 24)),
                1: list(range(24, 48)),
            }
            model.parallelize(device_map)
        elif num_devices == 1:
            model.cuda()
        else:
            raise ValueError(f'Num devices ({num_devices}) not supported for gpt2-xl.')
    else:
        model.cuda()
    return model 


def get_lm(args, num_devices=None, load_mode=None):
    if load_mode is None:
        model_name = args.model_name
    elif load_mode == 'guidance':
        model_name = args.guidance_model_name
    else:
        raise ValueError(f'Unknown load_mode: {load_mode}')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
    )
    if num_devices is not None:
        model = put_model_on_gpus(model, model_name, num_devices)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f'Model "{model_name}" loaded.')
    return model


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_node_children_in_text(text, node, hierarchy):
    """
    Check if `text` contains `node`'s children in the `hierarchy`.
    """
    nodes = set(hierarchy[node] + [node])
    return list(set([node for node in nodes if node in text]))


def save_ckpt(model, args, run_dir, **kwargs):
    stats = dict()
    stats['current_epoch'] = kwargs.get('epoch', None)
    stats['global_step'] = kwargs.get('global_step', None)
    stats['best_metric'] = kwargs.get('best_metric', None)

    with open(run_dir / 'stats.yaml', 'w') as f:
        yaml.dump(stats, f)

    args.model_class_name = model.__class__.__name__
    args.ckpt_path = run_dir / 'checkpoint.pt'
    checkpoint = dict()
    checkpoint['args'] = vars(args)
    checkpoint['states'] = model.state_dict()

    torch.save(checkpoint, args.ckpt_path)
    print(f"Model saved at: {args.ckpt_path}")


def load_ckpt(load_path):
    checkpoint = torch.load(load_path, map_location='cpu')
    args = Namespace(**checkpoint['args'])
    args = update_args(args)
    states = checkpoint['states']
    model_class = get_model_class(args.model_class_name)
    model = model_class(args)
    model.load_state_dict(states)
    model.eval()
    print('Model loaded from:', load_path)
    return model, args
