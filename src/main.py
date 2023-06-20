import re
import sys
import yaml
import json
import time
import random
import logging
import argparse
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from rich import print
from alive_progress import alive_bar

from src.utils import (
    get_data,
    get_lm,
    get_tokenizer,
    cleanup_gen_text,
    reformat_text,
    top_p_sampling,
)
from src.diverse_instructions import prepare_context
from src.guide import (
    run_pplm_step,
    run_wd_step,
    run_constrained_decoding,
)
from src.guide_with_offset import run_offset_step
from src.guidance_utils import add_key_values, get_guidance_model
from src.metrics import compute_prediction_metrics, aggregate_metrics
from src.lm_scorer import LMScorer

from src.experiment import ExperimentManager
from src.metric_tracker import MetricTracker

random.seed(0)


def run_generation(
    datapoint,
    hierarchy,
    model,
    tokenizer,
    args,
    guidance_model=None,
    in_guidance_model=None
):
    if args.eval_version == -2:
        context = datapoint['context']
    else:
        context = datapoint['context_with_instructions']
    #context = datapoint['context']
    context = tokenizer.encode(context, return_tensors='pt').cuda()
    start_context_length = context.size(1)
    topic = datapoint['topic']
    constraint = datapoint['constraint']
    example_id = datapoint['id']
    if in_guidance_model is None:
        in_guidance_model = guidance_model

    gen_step = 0
    context_length = 0
    generated_tokens = []
    end = []
    refinements = []
    
    while (
        context_length < args.max_gen_length + start_context_length and
        end != [' ==', '\n']
    ):
        last_token = context[:, -1:]
        curr_context = context[:, :-1]

        outputs = model(curr_context)
        history = outputs.past_key_values

        gen_state = dict(
            example_id=example_id,
            args=args,
            model=model,
            guidance_model=guidance_model,
            in_guidance_model=in_guidance_model,
            tokenizer=tokenizer,
            history=history,
            last_token=last_token,
            curr_context=curr_context,
            topic=topic,
            constraint=constraint,
            datapoint=datapoint,
        )
        if 'pplm' in args.guidance:
            final_logits, refinement = run_pplm_step(**gen_state)
            final_probs = F.softmax(final_logits, dim=-1)
        elif 'wd' in args.guidance:
            final_logits, refinement = run_wd_step(**gen_state)
            final_probs = F.softmax(final_logits, dim=-1)
        elif 'cd' in args.guidance:
            final_logits, refinement = run_constrained_decoding(**gen_state)
            final_probs = F.softmax(final_logits, dim=-1)
        elif args.guidance == 'none' or 'nl' in args.guidance:
            final_outputs = model(last_token, past_key_values=history)
            final_logits = final_outputs.logits[:, -1, :]
            final_probs = F.softmax(final_logits, dim=-1)
            refinement = []
        elif 'os' in args.guidance:
            final_probs, refinement = run_offset_step(**gen_state)
        else:
            raise ValueError(f'Guidance type `{args.guidance}` not supported.')
        refinements.append(refinement)

        if args.fusion_gamma is not None or args.fusion_gamma != 1.0:
            orig_outputs = model(last_token, past_key_values=history)
            orig_logits = orig_outputs.logits[:, -1, :]
            orig_probs = F.softmax(orig_logits, dim=-1)
            final_probs = final_probs.pow(args.fusion_gamma) * \
                          orig_probs.pow(1.0 - args.fusion_gamma)

        if args.top_p is not None:
            final_logits = top_p_sampling(final_logits, args.top_p)
            final_probs = F.softmax(final_logits / args.temperature, dim=-1)
            next_token = torch.multinomial(final_probs, num_samples=1)
        else:
            next_token = torch.argmax(final_probs, dim=1)
        next_token_text = tokenizer.decode(
            next_token,
            skip_special_tokens=True
        )
        
        context = torch.cat([context, next_token[:, None]], dim=-1)
        context_length = context.size(1)

        generated_tokens.append(next_token)
        gen_step += 1

        end.append(next_token_text)
        end = end[-2:]
    return generated_tokens, refinements


def setup_logger(args):
    handlers = []
    run_dir = Path(args.run_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / 'predictions.jsonl'
    config_path = run_dir / 'config.yaml'

    if args.override == 'manual':
        ans = input(
            f'The following files will be override:\n'
            f'`{prediction_path}`\n'
            f'`{config_path}`\n'
            f'Proceed? [yes/no]: '
        )
        if ans.lower() == 'yes':
            handlers.append(logging.FileHandler(prediction_path, 'w'))
            with open(config_path, 'w') as f:
                yaml.dump(vars(args), f)
        else:
            sys.exit(0)
    elif args.override == 'auto':
        handlers.append(logging.FileHandler(prediction_path, 'w'))
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)
    elif args.override == 'no':
        handlers.append(logging.FileHandler(prediction_path))
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)
    else:
        raise ValueError(f'`{args.override}` not recognized.')

    if args.log_to_console:
        handlers.append(RichHandler())

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        handlers=handlers
    )

    # Save command to file in `run_dir``.
    run_command = ' '.join(['python -m ci.main'] + sys.argv[1:])
    with open(run_dir / 'run.sh', 'w') as f:
        f.write(run_command)
    return run_dir


def setup_args(args):
    if args.guidance_model_name == 'oracle':
        args.max_multistep_len = 0
    if args.guidance != 'none':
       args.guidance = args.guidance.split('+')
    return args


def run(args, run_id, exp_manager=None):
    #run_dir = setup_logger(args)
    args = setup_args(args)

    vs = []
    os = []
    onvs = []

    num_devices = torch.cuda.device_count()

    datasets, hierarchy, gold = get_data(args)
    model = get_lm(args, num_devices)
    tokenizer = get_tokenizer(args)

    # Load the guidance model.
    if args.guidance_model_type != 'none':
        guidance_model = get_guidance_model(
            args,
            tokenizer,
            hierarchy,
            num_devices=num_devices
        )
    else:
        guidance_model = None

    if (args.guidance_model_type_2 != args.guidance_model_type and
        args.guidance_model_type_2 != 'none'):
        in_guidance_model = get_guidance_model(
            args,
            tokenizer,
            hierarchy,
            guidance_lm=guidance_model.guidance_lm
        )
    else:
        in_guidance_model = None

    lm_scoreer = LMScorer(model=model, tokenizer=tokenizer)

    # Main loop.
    predictions = []
    prediction_metrics = []
    total = len(datasets[args.dataset_split])
    
    with alive_bar(total, enrich_print=False) as bar:
        for idx, datapoint in enumerate(datasets[args.dataset_split]):
            if len(predictions) == args.num_datapoints:
                break

            topic = datapoint['topic']
            constraint = datapoint['constraint']

            if args.eval_version == -1 and args.dataset_split == 'dev':
                eval_version = random.choice(range(3, 6))
            elif args.eval_version == -1 and args.dataset_split == 'test':
                eval_version = random.choice(range(6, 35))
            else:
                eval_version = args.eval_version
            
            datapoint = prepare_context(datapoint, args, version=eval_version)

            # NOTE: The main logic starts here.
            gen_ids, refinements = run_generation(
                datapoint,
                hierarchy,
                model,
                tokenizer,
                args,
                guidance_model,
                in_guidance_model=in_guidance_model,
            )

            generated_text = tokenizer.decode(
                torch.cat(gen_ids),
                skip_special_tokens=True
            )
            generated_text = cleanup_gen_text(generated_text)
            prediction = dict(
                datapoint=datapoint,
                guidance=(
                    '+'.join(args.guidance)
                    if args.guidance != 'none' else args.guidance
                ),
                generated_text=generated_text,
                refinements=refinements,
                generated_tokens=[tokenizer.decode(token) for token in gen_ids],
            )
            predictions.append(prediction)

            prediction_metric_outputs = compute_prediction_metrics(
                prediction,
                hierarchy,
                data_mode=args.data,
            )
            prediction_metric = prediction_metric_outputs['prediction_metric']
            prediction_metric['id'] = datapoint['id']
            extracted = prediction_metric_outputs['extracted']
            if generated_text:
                ppl = lm_scoreer.sentence_score(generated_text)
            else:
                ppl = 0.0

            prediction_metric.update({'ppl': ppl})   
            prediction_metrics.append(prediction_metric)

            exp_manager.take(
                {**prediction, **prediction_metric},
                log=True,
                console=False,
                run_id=run_id,
                add_metric=True,
            )

            v = prediction_metric['violated']
            o = prediction_metric['on_topic']
            onv = prediction_metric['on_topic_not_violated']
            vs.append(v)
            os.append(o)
            onvs.append(onv)
            print(f'name: {args.name} (num={len(predictions)})')
            print(args)
            print(f'eval_version: {eval_version}')
            print(f'topic: {topic}')
            print(datapoint['context'])
            print(f'constraint: {constraint}')
            print(f'generated_text:')
            print(generated_text)
            print(f'Extracted: {extracted}')
            print(f'v: {v}')
            print(f'o: {o}')
            print(f'onv: {onv}')
            print(f'ppl: {ppl:.4f}')
            print(f'Accumulated violated: {sum(vs) / len(vs):.4f}')
            print(f'Accumulated on_topic: {sum(os) / len(os):.4f}')
            print(f'Accumulated on_topic_not_violated: {sum(onvs) / len(onvs):.4f}')
            print('---')

            time.sleep(0.005)
            bar()
    
    stats = exp_manager.aggregate_metrics()
    exp_manager.save_results(stats)
    print(stats)
    print(args)
    print('Run finished.')


def get_argparse(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="./runs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--override", type=str, default="manual",
        choices=['auto', 'manual', 'no'],
        help=(
            "`auto`: override all, "
            "`manual`: ask in terminal before override, "
            "`no`: no override."
        )
    )
    parser.add_argument("--data", type=str, default='wordnet',
        choices=['wordnet', 'wikidata'],
    )

    # Data arguments.
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--dev_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--hierarchy_path", type=str, default=None)
    parser.add_argument("--wiki_gold", type=str, default=None)
    parser.add_argument("--num_datapoints", type=int, default=500)
    parser.add_argument("--dataset_split", type=str, default='dev')
    parser.add_argument("--eval_version", type=int, default=0,
        help=(
            "-1: use dataset_split to decide which version to use.\n"
            "-2: use only the context."
        )
    )

    # Generation model arguments.
    parser.add_argument("--model_name", type=str, default="gpt2-xl")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance", type=str, default='none',
        choices=[
            'none',                   # No constraint
            'pplm+ex', 'pplm+ex+in',  # PPLM
            'wd+ex', 'wd+ex+in',      # Weighted decoding
            'cd+ex',                  # Constrained decoding
            'nl+ex', 'nl+ex+in',      # Natural language constraint
            'os+ex+in',               # Offset guidance
        ],
    )
    parser.add_argument("--max_gen_length", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=None)  # 0.02 / 0.1
    parser.add_argument("--beta", type=float, default=None)  # 0.05
    parser.add_argument("--gamma", type=float, default=None)  # 1.5
    parser.add_argument("--fusion_gamma", type=float, default=1.0)  # 1.0
    parser.add_argument("--refinement_steps", type=int, default=0)  # 3
    parser.add_argument("--prev_run_dir", type=str, default=None)
    parser.add_argument("--log_to_console", action="store_true")

    # Guidance model.
    parser.add_argument("--guidance_model_name", type=str, default='none',
        choices=[
            'none',
            'oracle',
            'gpt2', 'gpt2-xl', 'gpt2-ft',
            'EleutherAI/gpt-j-6B',
        ],
        help="Guidance model. Default is the oracle."
    )
    parser.add_argument("--guidance_model_type", type=str, default='none',
        choices=['none', 'full', 'binary', 'discrete'],
        help="Guidance model type."
    )
    parser.add_argument("--guidance_model_type_2", type=str, default='none',
        choices=['none', 'full', 'binary', 'discrete'],
        help="Second guidance model (will only be used for INCLUSION if specified)."
    )
    parser.add_argument("--guidance_model_path", type=str, default="",
        help="Load checkpointed guidance model."
    )
    parser.add_argument("--num_icl_pairs", type=int, default=0, help="Default: 0")
    parser.add_argument("--g_threshold", type=float, default=0.5, help="Default: 0.5")
    parser.add_argument("--max_multistep_len", type=int, default=0, help="Default: 8")
    parser.add_argument("--full_guide_topk_in", type=int, default=0, help="Default: 40")
    parser.add_argument("--full_guide_topk_ex", type=int, default=0, help="Default: 20")
    parser.add_argument("--discrete_max_length", type=int, default=100,
        help="Default: 100. The number of tokens generated for discrete guidance."
    )
    parser.add_argument("--discrete_guidance_num_beams", type=int, default=1)
    parser.add_argument("--discrete_guidance_num_beam_groups", type=int, default=1)
    parser.add_argument("--discrete_guidance_do_sample", type=bool, default=False)
    parser.add_argument("--discrete_guidance_top_k", type=int, default=None)
    parser.add_argument("--discrete_guidance_top_p", type=float, default=None)
    parser.add_argument("--discrete_guidance_temperature", type=float, default=None)
    parser.add_argument(
        "--discrete_guidance_num_return_sequences", type=int, default=None
    )
    parser.add_argument("--discrete_guidance_diversity_penalty", type=float, default=None)
    parser.add_argument(
        "--discrete_guidance_instruct2guide_model_dir", type=str, default=None,
        help="Directory of tuned prefixes for topic and constraint."
    )
    parser.add_argument("--discrete_guidance_use_trie", action="store_true")

    if config is not None and isinstance(config, dict):
        parser.set_defaults(**config)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    python -m src.main --name RUN_NAME
    """
    args = get_argparse()

    exp_manager = ExperimentManager(
        name=args.name,
        run_dir=args.run_dir,
        override=True,
        num_runs=1,
        metric_tracker=MetricTracker(),
    )
    run(args, run_id=0, exp_manager=exp_manager)

