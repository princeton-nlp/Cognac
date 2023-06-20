"""
The guidance procedure.
- No Guidance
- PPLM (pplm)
- Weighted Decoding (wd)
"""
import torch
import torch.nn.functional as F

from src.guidance_utils import (
    get_gradient,
    get_query_entity,
    deepcopy_history,
    add_key_values,
)


def run_pplm_step(
    example_id,
    args,
    model,
    guidance_model,
    in_guidance_model,
    tokenizer,
    history,
    last_token,
    curr_context,
    topic,
    constraint,
    datapoint,
):
    """
    Perform PPLM refinement steps to obtain the perturbed history.
    """
    history_delta = [
        (torch.zeros_like(key), torch.zeros_like(value))
        for key, value in history
    ]
    refinement = []  # Track refinement tokens and the guidance probabilties.
    for i in range(args.refinement_steps):
        curr_history_delta = [
            (
                key.clone().detach().requires_grad_(True),
                value.clone().detach().requires_grad_(True)
            ) 
            for key, value in history_delta
        ]
        perturbed_history = add_key_values(history, curr_history_delta)

        # Generate multiple tokens to query the guidance model.
        next_token_ = last_token.clone().detach()
        history_ = deepcopy_history(perturbed_history)
        multistep_tokens = []
        multistep_probs = []
        for _ in range(args.max_multistep_len):
            outputs_ = model(next_token_, past_key_values=history_)
            logits_ = outputs_.logits[:, -1, :]
            probs_ = F.softmax(logits_, dim=-1)
            history_ = outputs_.past_key_values

            next_token_ = torch.argmax(probs_, dim=1).unsqueeze(0)
            multistep_tokens.append(next_token_.item())
            multistep_probs.append(probs_)
        
        # Use the list of probs/histories of the entity tokens for further use.
        query_entity, query_tags, parsed = \
            get_query_entity(multistep_tokens, tokenizer)

        # Still use the latest token for generatation while useing the
        # full entity probs to get the guidance signal.
        next_token_after_refinement = multistep_tokens[0]
        next_token_after_refinement_text = tokenizer.decode(
            next_token_after_refinement,
            skip_special_tokens=True,
        ).strip(' ').replace('\n', '\\n')
        perturbed_probs = multistep_probs[0]

        ex_loss = 0.0
        ex_grad = None
        ex_guidance_outs = None
        if 'ex' in args.guidance:
            ex_guidance_outs = guidance_model.calc_loss(
                perturbed_probs,
                query_entity,
                constraint,
                mode='ex',
                threshold=args.g_threshold,
            )
            ex_loss = ex_guidance_outs['loss']

            # NOTE: the sign of the loss is assigned here
            ex_grad = get_gradient(
                ex_loss,
                curr_history_delta,
                args.alpha,
                args,
                retain_graph=True,
            )

        in_loss = 0.0
        in_grad = None
        in_guidance_outs = None
        if 'in' in args.guidance:
            in_guidance_outs = in_guidance_model.calc_loss(
                perturbed_probs,
                query_entity,
                topic,
                mode='in',
                threshold=args.g_threshold,
            )
            in_loss = in_guidance_outs['loss']

            # NOTE: the sign of the loss is assigned here
            in_grad = get_gradient(
                -in_loss,
                curr_history_delta,
                args.beta,
                args,
                retain_graph=False,
            )

        grad = None
        if ex_grad is not None:
            grad = ex_grad
        if in_grad is not None:
            grad = add_key_values(in_grad, ex_grad)
        
        history_delta = add_key_values(history_delta, grad)
        for key, value in curr_history_delta:
            if key.grad is not None:
                key.grad.zero_()
            if value.grad is not None:
                value.grad.zero_()

        refinement.append(dict(
            last_token=tokenizer.decode(
                    last_token.item(),
                    skip_special_tokens=True
                ).strip(' ').replace('\n', '\\n'),
            next_token_after_refinement_text=next_token_after_refinement_text,
            query_entity=query_entity,
            query_tags=query_tags,
            parsed=parsed,
            ex_yes_prob=(
                ex_guidance_outs['yes_prob']
                if ex_guidance_outs is not None else -1.0
            ),
            ex_no_prob=(
                ex_guidance_outs['no_prob']
                if ex_guidance_outs is not None else -1.0
            ),
            in_yes_prob=(
                in_guidance_outs['yes_prob']
                if in_guidance_outs is not None else -1.0
            ),
            in_no_prob=(
                in_guidance_outs['no_prob']
                if in_guidance_outs is not None else -1.0
            ),
        ))
    perturbed_history = add_key_values(history, history_delta)
    final_outputs = model(last_token, past_key_values=perturbed_history)

    final_logits = final_outputs.logits[:, -1, :]
    return final_logits, refinement


def run_wd_step(
    example_id,
    args,
    model,
    guidance_model,
    in_guidance_model,
    tokenizer,
    history,
    last_token,
    curr_context,
    topic,
    constraint,
    datapoint,
):
    stepwise_info = []

    orig_outputs = model(last_token, past_key_values=history)
    orig_logits = orig_outputs.logits[:, -1, :]
    orig_probs = F.softmax(orig_logits, dim=-1)

    # Generate multiple tokens to query the guidance model.
    if not args.max_multistep_len:
        query_entity = None
    else:
        next_token_ = last_token.clone().detach()
        history_ = deepcopy_history(history)
        multistep_tokens = []
        for _ in range(args.max_multistep_len):
            outputs_ = model(next_token_, past_key_values=history_)
            logits_ = outputs_.logits[:, -1, :]
            probs_ = F.softmax(logits_, dim=-1)
            history_ = outputs_.past_key_values
            next_token_ = torch.argmax(probs_, dim=1).unsqueeze(0)
            multistep_tokens.append(next_token_.item())
    
        # Use the list of probs/histories of the entity tokens for further use.
        query_entity, query_tags, parsed = get_query_entity(multistep_tokens, tokenizer)
        if query_entity is None:
            return orig_logits, []

    wd_guidance_args = dict(
        example_id=example_id,
        orig_probs=orig_probs,
        query_entity=query_entity,
        threshold=args.g_threshold,
        curr_context=curr_context,
        last_token=last_token,
        datapoint=datapoint,
    )

    ex_inds, _, ex_info = guidance_model.query_guidance(
        constraint,
        mode='ex',
        **wd_guidance_args,
    )
    orig_logits[:, ex_inds] = orig_logits[:, ex_inds] - args.alpha

    if 'in' in args.guidance:
        in_inds, _, in_info = \
            in_guidance_model.query_guidance(topic, mode='in', **wd_guidance_args)
        orig_logits[:, in_inds] = orig_logits[:, in_inds] + args.beta
    else:
        in_info = dict()

    stepwise_info.append(dict(
        exclusion_info=ex_info,
        inclusion_info=in_info,
    ))
    final_logits = orig_logits
    return final_logits, stepwise_info


def run_constrained_decoding(
    example_id,
    args,
    model,
    guidance_model,
    in_guidance_model,
    tokenizer,
    history,
    last_token,
    curr_context,
    topic,
    constraint,
    datapoint,
):
    orig_outputs = model(last_token, past_key_values=history)
    orig_logits = orig_outputs.logits[:, -1, :]

    if 'ex' in args.guidance:
        words = guidance_model.hierarchy[constraint]
        if args.data == 'wordnet':
            words = words + [constraint]
        elif args.data == 'wikidata':
            raise ValueError('Not finished yet. To be completed.')
        else:
            raise ValueError(f'Data arg {args.data} not recognized.')
        
        bow_indices = [
            tokenizer.encode(word.strip(), add_prefix_space=True)
            for word in words
        ]
        for inds in bow_indices:
            orig_logits[:, inds] = float('-inf')

    return final_logits, []
