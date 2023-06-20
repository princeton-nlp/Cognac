"""
Offset Guidance.

Guide generation using the combination of three probabilities
1. p_1 = p(x_t | x_{<t})
2. p_2 = p(x_t | x_{<t}, topic)
2. p_3 = p(x_t | x_{<t}, constraint)

p(x_t | x_{<t}, topic, constraint) = \lambda_1 * p_1 + \lambda_2 * p_2 + \lambda_3 * p_3
"""

import torch
import torch.nn.functional as F

from src.utils import get_wikidata_p_text

t_history = None
c_history = None

SELF_DEBIASING_TEMPLATE = 'The following text contains examples of {}:'


@torch.no_grad()
def run_offset_step(
    example_id,
    args,
    model,
    tokenizer,
    history,
    last_token,
    curr_context,
    topic,
    constraint,
    datapoint,
    **kwargs
):
    global t_history, c_history

    if args.data == 'wikidata':
        topic = get_wikidata_p_text(topic[0], topic[1])
        constraint = get_wikidata_p_text(constraint[0], constraint[1])
    
    topic_prompt = SELF_DEBIASING_TEMPLATE.format(topic)
    constraint_prompt = SELF_DEBIASING_TEMPLATE.format(constraint)

    orig_outputs = model(last_token, past_key_values=history)
    orig_logits = orig_outputs.logits[:, -1, :]
#    orig_probs = F.softmax(orig_logits, dim=-1)

    if t_history is None:
        topic_prompt_ids = tokenizer.encode(topic_prompt, return_tensors='pt').cuda()
        t_outputs = model(topic_prompt_ids)
    else:
        t_outputs = model(last_token, past_key_values=t_history)
    t_history = t_outputs.past_key_values
    t_logits = t_outputs.logits[:, -1, :]
#    t_probs = F.softmax(t_logits, dim=-1)

    if c_history is None:
        constraint_prompt_ids = tokenizer.encode(constraint_prompt, return_tensors='pt').cuda()
        c_outputs = model(constraint_prompt_ids)
    else:
        c_outputs = model(last_token, past_key_values=c_history)
    c_history = c_outputs.past_key_values
    c_logits = c_outputs.logits[:, -1, :]
#    c_probs = F.softmax(c_logits, dim=-1)

#    final_probs = orig_probs + t_probs - c_probs
    alpha = 0.5
    beta = 0.5
    final_logits = orig_logits + beta * t_logits - alpha * c_logits
    final_probs = F.softmax(final_logits, dim=-1)

    t_history = None
    c_history = None
    return final_probs, []
