import re
import random
from pathlib import Path
from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk.tokenize import sent_tokenize

from src.guidance_utils import (
    remove_parenthesis,
    get_wikidata_query,
    get_wordnet_query,
)
from src.instruct2guide.utils import postprocess

_YES = 3763  # gpt2 tokenizer
_NO = 645  # gpt2 tokenizer
_PAD = 50256  # gpt2 tokenizer

random.seed(0)


class BaseGuidanceModel(nn.Module):
    def __init__(self, guidance_lm, tokenizer, args, **kwargs):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.guidance_lm = guidance_lm
        self.hierarchy = kwargs.get('hierarchy', None)
        self.entity_to_trie = dict()

        # Track generation state and use the next token for guidance.
        self.guidance_prefix = {
            'in': [],
            'ex': [],
        }

        # Track which `example_id` is done and stop guidance.
        self.in_finished = set()
    
    def forward(self, entity, query_entity=None, **kwargs):
        if self.args.data == 'wordnet':
            query = get_wordnet_query(
                guidance_model_type=self.guidance_model_type,
                entity=entity,
                query_entity=query_entity,
                args=self.args,
                **kwargs,
            )
        elif self.args.data == 'wikidata':
            query = get_wikidata_query(
                guidance_model_type=self.guidance_model_type,
                entity=entity,
                query_entity=query_entity,
                args=self.args,
                **kwargs,
            )
        else:
            raise ValueError(f'Unknown data: {self.args.data}')

        query_tokens = self.tokenizer.encode(query)
        query_tokens = torch.tensor(query_tokens).long().unsqueeze(0).cuda()
        out = self.guidance_lm(query_tokens)
        return out.logits[:, -1, :]
    
    def query_guidance(self, entity, **kwargs):
        """
        Returns indicies, loss, and info.
        """
        raise NotImplementedError

    def get_trie_filtered_indicies(
        self,
        mode,
        words,
        entity,
        last_token,
        curr_context,
        example_id,
    ):
        if self.args.data == 'wikidata':
            entity = tuple(entity)
        # Debug ########################################################
        #print('id:', example_id)
        #print('mode:', mode)
        #print('gen_examples:', words)
        #print('entity:', entity)
        #print('in_finished:', self.in_finished)
        ################################################################
        if mode == 'in' and example_id in self.in_finished:
            return []

        if entity not in self.entity_to_trie:
            trie = Trie()
            for word in words:
                name_tok_inds = self.tokenizer.encode(' ' + word)
                name_toks = [
                    self.tokenizer.decode(ind, skip_special_tokens=True)
                    for ind in name_tok_inds
                ]

                trie.start_node_inds.add(name_tok_inds[0])
                trie.insert(name_toks)
                trie.node_set.update(name_toks)
            self.entity_to_trie[entity] = trie

        # Get the trie corresponding to the current entity.
        trie = self.entity_to_trie[entity]
        last_token_text = self.tokenizer.decode(last_token.squeeze(0))

        # Debug #############################################################
        #curr_context_text = self.tokenizer.decode(curr_context.squeeze(0))
        #print('last:', last_token_text)
        #print('curr context:', curr_context_text)
        #print('node_set:', trie.node_set)
        #print(f'gp before [{mode}]: {self.guidance_prefix}')
        #####################################################################

        if last_token_text not in trie.node_set:
            # Debug ##########################################
            #print(f'[{mode}] here 1 -> [{last_token_text}]')
            ##################################################
            # Use the the start tokens for guidance.
            inds = list(trie.start_node_inds)
        else:
            # Debug ##########################################
            #print(f'[{mode}] here 2 -> [{last_token_text}]')
            ##################################################
            self.guidance_prefix[mode].append(last_token_text)
            query_prefix = self.guidance_prefix[mode]
            name_next_tok_index = len(query_prefix)

            name_candidates = trie.query(query_prefix)
            #print(f'Name candidates [{mode}]: {name_candidates}')
            if not name_candidates:
                # The prefix fails. Reset the tracker.
                inds = list(trie.start_node_inds)
                self.guidance_prefix[mode] = []
            else:
                name_candidate_tok_texts = [
                    c[name_next_tok_index] for c in name_candidates 
                    if len(c) > name_next_tok_index
                ]
                #print(name_candidate_tok_texts)
                inds = [
                    self.tokenizer.encode(tok)[0] for tok in name_candidate_tok_texts
                ]
            name_candidates = {''.join(toks) for toks in name_candidates}
            hit = ''.join(query_prefix) in {''.join(toks) for toks in name_candidates}
            if hit:
                self.in_finished.add(example_id)
                self.guidance_prefix[mode] = []
        # Debug ######################################################################
        #print(f'gp after [{mode}]: {self.guidance_prefix}')
        #print([self.tokenizer.decode(ind) for ind in inds])
        #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #input()
        ##############################################################################
        return inds


class BinaryGuidanceModel(BaseGuidanceModel):
    def __init__(self, guidance_lm, tokenizer, args, **kwargs):
        super().__init__(guidance_lm, tokenizer, args, **kwargs)
        self.guidance_model_type = 'binary'
    
    def query_guidance(self, entity, **kwargs):
        mode = kwargs.get('mode', None)
        query_entity = kwargs['query_entity']
        orig_probs = kwargs['orig_probs']
        if query_entity is None:
            return [], None, dict(
                yes_prob=-1.0,
                no_prob=-1.0,
            )
        if self.args.guidance_model_name == 'oracle':
            if self.args.data == 'wordnet':
                oracle_words = self.hierarchy[entity] + [entity]
                oracle_plural_words = [w + 's' for w in oracle_words]
                all_oracle_words = set(oracle_words) | set(oracle_plural_words)
                normalized_yes_prob = 1.0 * (query_entity in all_oracle_words)
            elif self.args.data == 'wikidata':
                words = self.hierarchy[entity]
                words = [remove_parenthesis(title) for q_id, title in words]
                words = set([w.lower() for w in words])
                normalized_yes_prob = 1.0 * (query_entity in words)
            else:
                raise ValueError(f'Data arg {self.args.data} not recognized.')
            yes_prob = normalized_yes_prob
            no_prob = 1.0 - normalized_yes_prob            
        else:
            logits = self.forward(entity, query_entity)
            probs = F.softmax(logits, dim=-1)
            yes_prob = probs[:, _YES].squeeze(0).item()
            no_prob = probs[:, _NO].squeeze(0).item()
            normalized_yes_prob = yes_prob / (yes_prob + no_prob)

        if normalized_yes_prob > kwargs['threshold']:
            inds = torch.argmax(orig_probs, dim=1)
            loss = torch.log(orig_probs.max())
        else:
            loss = None
            inds = []

        return inds, loss, dict(
            yes_prob=yes_prob,
            no_prob=no_prob,
            query_entity=query_entity,
            entity=entity,
        )


class FullGuidanceModel(BaseGuidanceModel):
    """
    Full guidance model operates with the next token's full probability,
    as opposed to binary guidance model only offers yes/no probability.
    """
    def __init__(self, guidance_lm, tokenizer, args, **kwargs):
        super().__init__(guidance_lm, tokenizer, args, **kwargs)
        self.guidance_model_type = 'full'
        self.in_bow_vecs = None
        self.ex_bow_vecs = None
        self.in_set = None
        self.ex_set = None

    def query_guidance(self, entity, **kwargs):
        mode = kwargs.get('mode', None)
        query_entity = kwargs['query_entity']
        orig_probs = kwargs['orig_probs']

        if self.args.guidance_model_name == 'oracle':
            if mode == 'in':
                self.in_bow_vecs, self.in_set = self._init_bow_vecs(entity, **kwargs)
                bow_vecs = self.in_bow_vecs
                ind_set = self.in_set
            elif mode == 'ex':
                self.ex_bow_vecs, self.ex_set = self._init_bow_vecs(entity, **kwargs)
                bow_vecs = self.ex_bow_vecs
                ind_set = self.ex_set
            else:
                raise ValueError(f'`mode` {mode} not supported.')
        
            loss = 0.0
            for vec in bow_vecs:
                loss += torch.log(torch.mm(orig_probs, torch.t(vec)).sum())
            inds = list(ind_set)
        else:
            logits = self.forward(entity, **kwargs)
            if mode == 'in':
                full_guide_topk = self.args.full_guide_topk_in
            elif mode == 'ex':
                full_guide_topk = self.args.full_guide_topk_ex
            else:
                raise ValueError(f'`mode` {mode} not supported.')

            _, inds = torch.topk(logits, k=full_guide_topk)
            inds = inds.squeeze(0)  # 1xk -> k
            topk_probs = orig_probs[:, inds]
            loss = torch.log(topk_probs.sum())

            ind_tokens = [self.tokenizer.decode(ind) for ind in inds]

        return inds, loss, dict(
            yes_prob=-1.0,
            no_prob=-1.0,
            query_entity=query_entity,
            entity=entity,
            ind_tokens=ind_tokens,
        )
    
    def _init_bow_vecs(self, entity, **kwargs):
        mode = kwargs.get('mode', None)
        curr_context = kwargs.get('curr_context', None)
        if self.args.data == 'wordnet':
            if mode == 'in':
                words = self.hierarchy[entity]
            elif mode == 'ex':
                words = self.hierarchy[entity] + [entity]
            words = words + [w + 's' for w in words]
        elif self.args.data == 'wikidata':
            if mode == 'in':
                words = self.hierarchy[entity][:100]
            elif mode == 'ex':
                words = self.hierarchy[entity]
            else:
                raise ValueError(f'`{mode}` not recognized.')

            # If full length is used there's gonna be too many Q's.
            words = [remove_parenthesis(q_title) for q_id, q_title in words]
        else:
            raise ValueError(f'Data arg {self.args.data} not recognized.')

        inds = self.get_trie_filtered_indicies(
            mode,
            words,
            entity, 
            kwargs['last_token'],
            kwargs['curr_context'],
            kwargs['example_id'],
        )
        
        vecs = []
        if inds:
            inds = torch.tensor(inds)
            onehot = torch.zeros(1, len(self.tokenizer))
            onehot[:, inds] = 1.0
            vecs.append(onehot.cuda())
        return vecs, set(inds)


class DiscreteGuidanceModel(BaseGuidanceModel):
    def __init__(self, guidance_lm, tokenizer, args, **kwargs):
        super().__init__(guidance_lm, tokenizer, args, **kwargs)
        self.guidance_model_type = 'discrete'
        # NOTE: this might cause memory leak. This might be largely fine 
        # given the expected number of new entity.
        self.enity_to_examples = dict()

    def forward_prefix(self, entity, **kwargs):
        datapoint = kwargs['datapoint']
        if self.args.data == 'wikidata':
            entity = tuple(entity)
        #gen_examples = self.enity_to_examples.get(entity, None)
        gen_examples = self.enity_to_examples.get((entity, datapoint['version']), None)
        if gen_examples is not None:
            return dict(gen_examples=gen_examples)

        datapoint = kwargs['datapoint']
        text = datapoint['context_with_instructions'] + ' [SEP]'

        if kwargs['mode'] == 'in':
            mode = 'topic'
        elif kwargs['mode'] == 'ex':
            mode = 'constraint'
        else:
            raise ValueError(f'mode={mode} not recognized.')
        
        gen_text = self.guidance_lm.generate(text, mode=mode)
        gen_examples = postprocess(gen_text, data_mode=self.args.data)
        #self.enity_to_examples[entity] = gen_examples
        self.enity_to_examples[(entity, datapoint['version'])] = gen_examples
        return dict(
            gen_examples=gen_examples,
            gen_texts=[gen_text],
        )

    def forward(self, entity, **kwargs):
        datapoint = kwargs['datapoint']
        if self.args.data == 'wikidata':
            entity = tuple(entity)
        #gen_examples = self.enity_to_examples.get(entity, None)
        gen_examples = self.enity_to_examples.get((entity, datapoint['version']), None)
        if gen_examples is not None:
            return dict(gen_examples=gen_examples)

        if self.args.data == 'wordnet':
            query = get_wordnet_query(
                guidance_model_type=self.guidance_model_type,
                entity=entity,
                args=self.args,
                **kwargs,
            )
        elif self.args.data == 'wikidata':
            query = get_wikidata_query(
                guidance_model_type='discrete',
                entity=entity,
                **kwargs,
            )
        else:
            raise ValueError(f'{self.args.data} not recognized.')

        query_tokens = self.tokenizer.encode(query)
        query_tokens = torch.tensor(query_tokens).long().unsqueeze(0).cuda()

        outputs = self.guidance_lm.generate(
            query_tokens,
            max_length=self.args.discrete_max_length,
            #skip_special_tokens=True,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=self.args.discrete_guidance_do_sample,
            num_beams=self.args.discrete_guidance_num_beams,
            num_beam_groups=self.args.discrete_guidance_num_beam_groups,
            top_p=self.args.discrete_guidance_top_p,
            top_k=self.args.discrete_guidance_top_k,
            temperature=self.args.discrete_guidance_temperature,
            diversity_penalty=self.args.discrete_guidance_diversity_penalty,
            num_return_sequences=self.args.discrete_guidance_num_return_sequences or 1,
        )
        gen_examples = []
        gen_texts = []
        pattern = 'Some examples are:'
        for output in outputs:
            gen_text = self.tokenizer.decode(output, skip_special_tokens=True)
            gen_texts.append(gen_text)
            gen_sents = sent_tokenize(gen_text)
            gen_sents = [sent for sent in gen_sents if sent.startswith(pattern)]
            gen = gen_sents[1].replace(pattern, '').strip().strip('.').split(', ')
            gen_examples += gen
        gen_examples = list(set(gen_examples))
        #self.enity_to_examples[entity] = gen_examples
        self.enity_to_examples[(entity, datapoint['version'])] = gen_examples
        return dict(
            gen_examples=gen_examples,
            gen_texts=gen_texts,
        )
    
    def query_guidance_with_loss(self, entity, **kwargs):
        mode = kwargs.get('mode', None)
        gen_examples = self.forward(entity)

        if mode == 'in':
            self.in_bow_vecs, self.in_set = self._init_bow_vecs(gen_examples)
            bow_vecs = self.in_bow_vecs
        elif mode == 'ex':
            self.ex_bow_vecs, self.ex_set = self._init_bow_vecs(gen_examples)
            bow_vecs = self.ex_bow_vecs
        else:
            raise ValueError(f'`mode` {mode} not supported.')
        
        loss = 0.0
        for vec in bow_vecs:
            loss += torch.log(torch.mm(next_token_prob, torch.t(vec)).sum())

        return dict(
            loss=loss,
            yes_prob=-1.0,
            no_prob=-1.0,
            gen_examples=gen_examples,
            query_entity=query_entity,
            entity=entity,
        )
    
    def query_guidance(self, entity, **kwargs):
        mode = kwargs.get('mode', None)
        if self.args.discrete_guidance_instruct2guide_model_dir is not None:
            gen_out = self.forward_prefix(entity, **kwargs)
        else:
            gen_out = self.forward(entity, **kwargs)

        inds = self._init_bow_vecs(
            gen_out['gen_examples'],
            entity=entity,
            return_only_inds=True,
            **kwargs
        )
        inds = list(inds)
        gen_info = dict(
            gen_examples=gen_out['gen_examples'],
            entity=entity,
        )
        return inds, None, gen_info

    def _init_bow_vecs(self, words, entity, return_only_inds=False, **kwargs):
        if self.args.data == 'wordnet':
            words = words + [w + 's' for w in words]

        # Use trie.
        if self.args.discrete_guidance_use_trie:
            inds = self.get_trie_filtered_indicies(
                kwargs['mode'],
                words,
                entity, 
                kwargs['last_token'],
                kwargs['curr_context'],
                kwargs['example_id'],
            )
            if return_only_inds:
                return set(inds)
        else:
            bow_indices = [
                self.tokenizer.encode(word.strip(), add_prefix_space=True)
                for word in words
            ]
            bow_set = set([ind for inds in bow_indices for ind in inds])
            if return_only_inds:
                return bow_set


class OracleGuidanceModel(BaseGuidanceModel):
    def __init__(self, guidance_lm, tokenizer, args, **kwargs):
        super().__init__(guidance_lm, tokenizer, args, **kwargs)
        self.guidance_model_type = 'oracle'

    def query_guidance(self, entity, **kwargs):
        mode = kwargs.get('mode', None)

        if self.args.data == 'wordnet':
            if mode == 'in':
                words = self.hierarchy[entity]
            elif mode == 'ex':
                words = self.hierarchy[entity] + [entity]
            else:
                raise ValueError(f'`{mode}` not recognized.')
        elif self.args.data == 'wikidata':
            if mode == 'in':
                words = self.hierarchy[entity][:100]
            elif mode == 'ex':
                words = self.hierarchy[entity][:1000]
            else:
                raise ValueError(f'`{mode}` not recognized.')
            # If full length is used there's gonna be too many Q's.
            words = [remove_parenthesis(q_title) for q_id, q_title in words]
        else:
            raise ValueError(f'Data arg {self.args.data} not recognized.')

        inds = self._init_bow_vecs(
            words,
            entity=entity,
            return_only_inds=True,
            **kwargs
        )
        inds = list(inds)
        gen_info = dict(
            entity=entity,
        )
        #print(mode)
        #print(entity)
        #print(words)
        #print([self.tokenizer.decode(ind) for ind in inds])
        #print(self.guidance_prefix)
        #print('---')
        #input()
        return inds, None, gen_info

    def _init_bow_vecs(self, words, entity, return_only_inds=False, **kwargs):
        # Use trie.
        if self.args.discrete_guidance_use_trie:
            inds = self.get_trie_filtered_indicies(
                kwargs['mode'],
                words,
                entity, 
                kwargs['last_token'],
                kwargs['curr_context'],
                kwargs['example_id'],
            )
            if return_only_inds:
                return set(inds)
        else:
            bow_indices = [
                self.tokenizer.encode(word.strip(), add_prefix_space=True)
                for word in words
            ]
            bow_set = set([ind for inds in bow_indices for ind in inds])
            if return_only_inds:
                return bow_set


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, tok):
        self.tok = tok

        self.is_end = False

        # A counter indicating how many times a word 
        # is inserted (if this node's is_end is True).
        self.counter = 0

        # A dictionary of child nodes. Keys are tokens, values are nodes.
        self.children = {}


class Trie:
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode('')
        self.node_set = set()
        self.start_node_inds = set()
    
    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each token in the word.
        # Check if there is no child containing the character, 
        # create a new child for the current node.
        for tok in word:
            if tok in node.children:
                node = node.children[tok]
            else:
                # If a token is not found, create a new node in the trie.
                new_node = TrieNode(tok)
                node.children[tok] = new_node
                node = new_node
        
        node.is_end = True

        # Increment the counter to indicate that we see this word once more.
        node.counter += 1
        
    def dfs(self, node, prefix):
        """
        Depth-first traversal of the trie.
        
        Args:
            - node: the node to start with.
            - prefix: the current prefix, for tracing a word while traversing.
        """
        if node.is_end:
            self.output.append(prefix + [node.tok])
        
        for child in node.children.values():
            self.dfs(child, prefix + [node.tok])
        
    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of 
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # as there can be more than one word with such prefix.
        self.output = []
        node = self.root
        
        # Check if the prefix is in the trie.
        for tok in x:
            if tok in node.children:
                node = node.children[tok]
            else:
                return []
        
        # Traverse the trie to get all candidates.
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return.
        return self.output
