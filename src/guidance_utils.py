"""
Utilities relevant to the guidance models.
"""
import re
from pathlib import Path

import torch
import spacy

from src.utils import (
    to_namespace,
    get_data,
    get_lm,
    get_tokenizer,
    cleanup_gen_text,
    reformat_text,
    load_ckpt,
)

SMALL_CONST = 1e-8
PROMPT_TEXT = (
#    'List out some examples of us presidents. Some examples are: lincoln, obama, washignton.\n'
#    'List out some examples of color. Some examples are: red, blue, green.\n'
#    'List out some examples of cities. Some examples are: new york, oslo, tokyo.\n'
    'List out some examples of movie genres. Some examples are: drama, horror, action.\n'
)
PROMPT_TEXT_TEMPLATE = {
    'animal': 'List out some examples of animals. Some examples are: cat, dog, lion.\n',
    'food': 'List out some examples of food. Some examples are: pizza, burger, pasta.\n',
    'vehicle': 'List out some examples of vehicles. Some examples are: car, bus, train.\n',
    'art': 'List out some examples of art. Some examples are: painting, sculpture, drawing.\n',
    'sport': 'List out some examples of sports. Some examples are: soccer, basketball, tennis.\n',
}

spacy_parser = spacy.load("en_core_web_lg")


def remove_parenthesis(s):
    m = re.search(r'(.*) \(.*\)', s)
    if m is None:
        return s
    else:
        return m.group(1)


def get_wikidata_query(
    guidance_model_type,
    entity,
    query_entity=None,
    **kwargs
):
    p_name, p_value = entity

    if guidance_model_type == 'binary':
        if p_name == 'place_of_birth':
            prompt = 'Was {0} born in {1}? The answer is'
        elif p_name == 'place_of_death':
            prompt = 'Did {0} die in {1}? The answer is'
        elif p_name == 'occupation':
            prompt = 'Was {0} a {1}? The answer is'
        elif p_name == 'country_of_citizenship':
            prompt = 'Was {0} a citizen of {1}? The answer is'
        elif p_name == 'academic_degree':
            prompt = 'Did {0} hold a degree in {1}? The answer is'
        elif p_name == 'educated_at':
            prompt = 'Did {0} get the education at {1}? The answer is'
        else:
            raise ValueError(f'Property name `{p_name}` not recognized.')
        query = prompt.format(query_entity, p_value)
        return query

    elif guidance_model_type in ('full', 'discrete'):
        #QUERY_FORMAT = 'List out some famous names of {0}. Some examples are:'
        #QUERY_FORMAT = 'List out some famous full names of passed {0}. Some examples are:'
        #QUERY_FORMAT = 'List out some historical names of {0}. Some examples are:'
        QUERY_FORMAT = 'List out some famous names of dead {0}. Some examples are:'
        if p_name == 'place_of_birth':
            fill_text = f'people who were born in {p_value}'
            prompt_text = f'List out some famous names of dead people who has traveled to France. Some examples are: Ernest Hemingway, Miles Davis, Oscar Wilde.\n'
        elif p_name == 'place_of_death':
            fill_text = f'people who died in {p_value}'
            prompt_text = f'List out some famous names of dead people who has traveled to France. Some examples are: Ernest Hemingway, Miles Davis, Oscar Wilde.\n'
        elif p_name == 'occupation':
            fill_text = p_value
            prompt_text = f'List out some famous names of dead people who were tech CEOs. Some examples are: Steve Jobs, Mark Hurd, Bill Campbell.\n'
        elif p_name == 'country_of_citizenship':
            fill_text = f'people who are citizens of {p_value}'
            prompt_text = f'List out some famous names of dead people who has traveled to France. Some examples are: Ernest Hemingway, Miles Davis, Oscar Wilde.\n'
        elif p_name == 'academic_degree':
            fill_text = f'people who hold a degree in {p_value}'
            prompt_text = f'List out some famous names of dead people who has traveled to France. Some examples are: Ernest Hemingway, Miles Davis, Oscar Wilde.\n'
        elif p_name == 'educated_at':
            #fill_text = f'people who had their education at {p_value}'
            fill_text = f'people who were educatied at {p_value}'
            prompt_text = f'List out some famous names of dead people who has traveled to France. Some examples are: Ernest Hemingway, Miles Davis, Oscar Wilde.\n'
        else:
            raise ValueError(f'Property name `{p_name}` not recognized.')
        query = QUERY_FORMAT.format(fill_text)
        #prompt_text = kwargs.get('prompt_text', PROMPT_TEXT)
        return (prompt_text + query).strip()

    else:
        raise ValueError(f'Guidance model type {guidance_model_type} not recognized.')


def get_wordnet_query(
    guidance_model_type,
    entity,
    query_entity=None,
    **kwargs,
):
    args = kwargs.get('args', None)
    if guidance_model_type == 'binary':
        MAGIC_PROMPT = 'I am an expert in taxonomy.'
        QUERY_FORMAT = 'Is {0} a type of {1}? The answer is{2}'
        query = [MAGIC_PROMPT]
        if args.num_icl_pairs > 0:
            pos_icl_pairs, neg_icl_pairs, noisy_queries = load_icl_pairs(args)
            icl_text = get_icl_query(
                args.num_icl_pairs,
                pos_icl_pairs,
                neg_icl_pairs,
                noisy_queries,
            )
        else:
            icl_text = []
        query += icl_text
        query.append(QUERY_FORMAT.format(query_entity, entity, ''))
        query = ' '.join(query).strip()
        return query
    
    elif guidance_model_type in ('full', 'discrete'):
        root_node = kwargs['datapoint']['parents'][0][0]
        prompt_text = PROMPT_TEXT_TEMPLATE.get(root_node, PROMPT_TEXT)
        
        QUERY_FORMAT = 'What are some examples of {0}? Some examples are:'
        query = QUERY_FORMAT.format(entity)
        return (prompt_text + query).strip()

    else:
        raise ValueError(f'Guidance model type {guidance_model_type} not recognized.')


def load_icl_pairs(args):
    """
    For in-context learning.
    """
    pos_icl_pairs = []
    neg_icl_pairs = []
    roots = ['animal', 'food', 'sport', 'art', 'vehicle']
    for root in roots:
        path = Path(args.icl_pair_dir) / f'train_pairs_{root}_new.txt'
        with open(path) as f:
            for line in f:
                cat, pos, neg = literal_eval(line.strip())
                cat = cat.replace('_', ' ')
                pos = pos.replace('_', ' ')
                neg = neg.replace('_', ' ')
                pos_icl_pairs.append((pos, cat, root))
                neg_icl_pairs.append((neg, cat, root))
    noisy_queries = []
    with open('/path/to/your/noisy_queries.txt') as f:
        for line in f:
            noisy_queries.append(line.strip())
    return pos_icl_pairs, neg_icl_pairs, noisy_queries


def get_icl_query(num_icl_pairs, pos_icl_pairs, neg_icl_pairs, noisy_queries):
    pos_icl_pairs = random.choices(pos_icl_pairs, k=num_icl_pairs)
    neg_icl_pairs = random.choices(neg_icl_pairs, k=num_icl_pairs)
    noisy_icl = random.choices(noisy_queries, k=num_icl_pairs)

    icl = []
    for p, n, noisy_q in zip(pos_icl_pairs, neg_icl_pairs, noisy_icl):
        icl += [
            QUERY_FORMAT.format(p[0], p[1], ' yes.'),
            QUERY_FORMAT.format(n[0], n[1], ' no.'),
#            QUERY_FORMAT.format(noisy_q, p[1], ' no.'),
#            QUERY_FORMAT.format(noisy_q, n[1], ' no.'),
        ]
    random.shuffle(icl)
    return icl


def get_guidance_model(
    args,
    tokenizer,
    hierarchy,
    num_devices=None,
    guidance_lm=None,
):
    from src.guidance_models import (
        BinaryGuidanceModel,
        FullGuidanceModel,
        DiscreteGuidanceModel,
        OracleGuidanceModel,
    )
    guidance_args = dict(
        tokenizer=tokenizer,
        args=args,
        hierarchy=hierarchy,
    )
    guidance_model_type = args.guidance_model_type

    # Load the core guidance LM (trained, un-trained, or oracle).
    if guidance_lm is not None:
        # Use the provided guidance_lm from the args.
        guidance_model_type = args.guidance_model_type_2
    elif args.guidance_model_path:
        # Load fine-tuned/prompt-tuned guidance models.
        guidance_lm, _ = load_ckpt(load_path=args.guidance_model_path)
    elif args.guidance_model_name == 'oracle':
        guidance_lm = None
    elif args.discrete_guidance_instruct2guide_model_dir is not None:
        assert args.guidance_model_type == 'discrete'
        from src.instruct2guide.utils import load_checkpoint
        guidance_lm  = load_checkpoint(
            args.discrete_guidance_instruct2guide_model_dir
        )
    else:
        guidance_lm = get_lm(args, num_devices, load_mode='guidance')

    # Construct the complete guidance model
    if args.guidance_model_path:
        guidance_model_class = BinaryGuidanceModel
    elif guidance_model_type == 'binary':
        guidance_model_class = BinaryGuidanceModel
    elif guidance_model_type == 'full':
        guidance_model_class = FullGuidanceModel
    elif guidance_model_type == 'discrete':
        guidance_model_class = DiscreteGuidanceModel
    elif guidance_model_type == 'oracle':
        guidance_model_class = OracleGuidanceModel
        guidance_lm = None
    else:
        raise ValueError(f'Guidance model type: {guidance_model_type} not supported.')

    guidance_args.update(guidance_lm=guidance_lm)
    guidance_model = guidance_model_class(**guidance_args)
    return guidance_model


def get_gradient(loss, curr_history_delta, step_size, args, retain_graph):
    if not torch.is_tensor(loss):
        # In the case where no loss is returned, we don't need to compute 
        # the gradient. This happens when using the guidance model and it
        # predicts that the answer to the  query is ``no''.
        grad = [
            (torch.zeros_like(key), torch.zeros_like(value))
            for key, value in curr_history_delta
        ]
    else:
        loss.backward(retain_graph=retain_graph)
        grad_norms = [
            (
                torch.norm(key.grad) + SMALL_CONST,
                torch.norm(value.grad) + SMALL_CONST
            )
            for key, value in curr_history_delta
        ]
        grad = [(
            - step_size * (key.grad / key_grad_norm ** args.gamma), 
            - step_size * (value.grad / value_grad_norm ** args.gamma)
        )
            for (key, value), (key_grad_norm, value_grad_norm) 
            in zip(curr_history_delta, grad_norms)
        ]
    return grad


def get_query_entity(tokens, tokenizer):
    """
    Check and return word/entity with the following order:
    1) tokens form a named entity + extra tokens
    2) tokens form a single word + extra tokens

    """
    text = tokenizer.decode(tokens).strip(' ')
    parsed = spacy_parser(text)
    parsed = [(t.text, t.pos_) for t in parsed]

    consecutive_idx = 0
    query_entity = []
    query_tags = []
    for i, (t_text, t_pos) in enumerate(parsed):
        if i == consecutive_idx and t_pos == 'PROPN':
            query_entity.append(t_text)
            query_tags.append(t_pos)
            consecutive_idx += 1

    if not query_entity and parsed and parsed[0][1] == 'NOUN':
        query_entity = [parsed[0][0]]
        query_tags = [parsed[0][1]]

    query_entity = ' '.join(query_entity)
    query_entity = query_entity.lower() if query_entity else None
    return query_entity, query_tags, parsed


def deepcopy_history(history):
    return [
        (key.clone(), value.clone())
        for key, value in history
    ]


def add_key_values(xs, deltas):
    added_xs = []
    for x, delta in zip(xs, deltas):
        x_k, x_v = x
        delta_k, delta_v = delta
        added_xs.append((x_k + delta_k, x_v + delta_v))
    return added_xs
