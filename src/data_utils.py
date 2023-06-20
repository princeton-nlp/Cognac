"""
"""
import json
import random
from collections import defaultdict

from tqdm import tqdm
from rich import print

import torch
from nltk.tokenize import sent_tokenize


def get_hierarchy(path):
    with open(path) as f:
        hierarchy_ = json.load(f)
    hierarchy = dict()
    for topic, leafs in hierarchy_.items():
        new_topic = topic.replace('_', ' ')
        new_leafs = [l.replace('_', ' ') for l in leafs]
        hierarchy[new_topic] = new_leafs
    return hierarchy


def load_datasets(data_paths, hierarchy, args):
    def process(datapoints, data_path):
        if 'pairs' in dataset_path:
            pass
        else:
            datapoints = [
                d for d in datapoints
                if hierarchy is not None and 
                d['constraint'] in hierarchy[d['topic']] and 
                d['constraint'] != d['topic']
            ]
        return datapoints

    datasets = dict()
    for dataset_name, dataset_path in data_paths.items():
        if dataset_path is not None:
            datapoints = []
            with open(dataset_path) as f:
                for line in f:
                    datapoint = json.loads(line.strip())
                    datapoints.append(datapoint)
            datapoints = process(datapoints, dataset_path)

        if args.randomize_dataset:
            random.seed(1)
            random.shuffle(datapoints)
        datasets[dataset_name] = datapoints
    return datasets


def get_guidance_data(args):
    if args.hierarchy_path is not None:
        hierarchy = get_hierarchy(args.hierarchy_path)
    else:
        hierarchy = None

    data_paths = {
        'train': args.train_path,
        'dev': args.dev_path,
        'test': args.test_path,
    }
    datasets = load_datasets(data_paths, hierarchy, args)
    
    if args.wiki_gold is not None:
        gold = dict()
        with open('data/wordnet_wiki_clean.jsonl') as f:
            for line in f:
                obj = json.loads(line.strip())
                gold[obj['node'].replace('_', ' ')] = obj
    else:
        gold = None

    print(f'Dataset loaded.')
    return datasets, hierarchy, gold


def get_bow(words, tokenizer):
    inds = []
    for word in words:
        inds += tokenizer.encode(word, add_prefix_space=True)
    inds = list(set(inds))

    inds = torch.tensor(inds)
    bow = torch.zeros(1, len(tokenizer))
    bow[:, inds] = 1.0

    return bow


def regressor_batcher(datapoints, batch_size, max_length, **kwargs):
    hierarchy = kwargs.get('hierarchy', None)
    tokenizer = kwargs.get('tokenizer', None)
    use_cuda = kwargs.get('use_cuda', False)
    num_batches = len(datapoints) // batch_size + 1

    for i in range(num_batches):
        batch = datapoints[i * batch_size:(i + 1) * batch_size]
        
        if not batch:
            continue
        
        topic_texts = [f"Talk about {b['topic']}" for b in batch]
        topic_inputs = tokenizer.batch_encode_plus(
            topic_texts,
            padding='longest'
        )
        constraint_texts = [f"Don't talk about {b['constraint']}" for b in batch]
        constraint_inputs = tokenizer.batch_encode_plus(
            constraint_texts,
            padding='longest'
        )
    
        topics = [b['topic'] for b in batch]
        constraints = [b['constraint'] for b in batch]
        topic_targets = torch.cat([
            get_bow(hierarchy.get(topic, []) + [topic], tokenizer)
            for topic in topics
        ], dim=0)
        constraint_targets = torch.cat([
            get_bow(hierarchy.get(constraint, []) + [constraint], tokenizer)
            for constraint in constraints
        ], dim=0)

        if use_cuda:
            topic_targets = topic_targets.cuda()
            constraint_targets = constraint_targets.cuda()

        yield dict(
            ids=[b['id'] for b in batch],
            topic_inputs=topic_inputs,
            constraint_inputs=constraint_inputs,
            topic_targets=topic_targets,
            constraint_targets=constraint_targets,
            topics=topics,
            constraints=constraints,
        )


def classifier_batcher(datapoints, batch_size, max_length, **kwargs):
    hierarchy = kwargs.get('hierarchy', None)
    tokenizer = kwargs.get('tokenizer', None)
    args = kwargs.get('args', None)
    num_batches = len(datapoints) // batch_size + 1

    for i in range(num_batches):
        batch = datapoints[i * batch_size:(i + 1) * batch_size]
        if not batch:
            continue
        
        if args.model_type == 'classification':
            texts = [sent_tokenize(b['text'])[1] for b in batch]
            label_texts = [b['label'] for b in batch]
            labels = torch.tensor([1 if b['label'] == 'yes' else 0 for b in batch]).long()
        elif args.model_type in ('prompt-tune', 'fine-tune'):
            _YES = 3763
            _NO = 645
            texts = [b['text'] for b in batch]
            label_texts = [b['label'] for b in batch]
            labels = torch.tensor([_YES if b['label'] == 'yes' else _NO for b in batch]).long()
        else:
            raise ValueError(f'Unknown model type: {args.model_type}')
        categories = [b['category'] for b in batch]

        encoded = tokenizer.batch_encode_plus(
            texts,
            max_length=max_length,
            padding='max_length',
        )
        input_ids = torch.tensor(encoded['input_ids']).long()
        attention_mask = torch.tensor(encoded['attention_mask']).long()

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        labels = labels.cuda()

        yield dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts,
            categories=categories,
            labels=labels,
            label_texts=label_texts,
        )


class Node:
    def __init__(self, name, parent, children):
        assert isinstance(parent, Node) or parent is None
        assert isinstance(children, list)
        self.name = name
        self.parent = parent
        self.children = children
    
    def __repr__(self):
        #parent = self.parent.name if self.parent is not None else None
        #children = [c.name for c in self.children]
        return f'Node(name={self.name})'


class Hierarchy:
    def __init__(self):
        self.name_to_node = dict()
        self.load()

    def get_node(self, name):
        return self.name_to_node.get(name, None)

    def get_leafs(self, node):
        pass
    
    def attach_children_to_parent(self, parent_name, children_names):
        parent = self.get_node(parent_name)

        if parent is None:
            parent = Node(parent_name, None, [])
            self.name_to_node[parent_name] = parent

        for child_name in children_names:
            child = self.get_node(child_name)
            if child is None:
                child = Node(child_name, parent, [])
                self.name_to_node[child_name] = child
            elif child.parent is None:
                child.parent = parent
            else:
                continue
            parent.children.append(child)

    def load(self):
        with open('data/hierarchy_path_to_children.jsonl') as f:
            for line in f:
                obj = json.loads(line.strip())
                hierarchy_path = ['<root>'] + obj['hierarchy_path']
                for p, c in zip(hierarchy_path, hierarchy_path[1:]):
                    self.attach_children_to_parent(p, [c])
                parent = hierarchy_path[-1]
                children = obj['children']
                self.attach_children_to_parent(parent, children)


if __name__ == '__main__':
    """
    python -m ci.data_utils
    """
    from transformers import AutoTokenizer
    hierarchy = get_hierarchy('data/topic_to_leafs.json')
    tokenizer = AutoTokenizer.from_pretrained('gpt2-xl', use_fast=False)

    datapoints = []
    with open('data/gpt2-xl/wiki_id_ood/sample/train.jsonl') as f:
        for line in f:
            datapoint = json.loads(line.strip())
            datapoints.append(datapoint)
    print(len(datapoints))
    datapoints = add_bow_to_datapoint(datapoints, hierarchy, tokenizer)
    print(len(datapoints))
