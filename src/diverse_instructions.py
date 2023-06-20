"""
Topic/constraint descriptions as the instruction.
"""
import csv
import copy
from src.utils import get_wikidata_p_text

MAPPING = {
    -2: [
        'begin+end',
        'dummy topic {}',
        'dummy constraint {}',
        'dummy',
    ]
}
with open('src/diverse_instructions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        index = int(row['index'])
        topic_inst = row['topic'].strip()
        constraint_inst = row['constraint'].strip()
        inst_type = row['type']
        mode = row['mode']

        if inst_type == 'begin+end':
            MAPPING[int(row['index'])] = [
                inst_type,
                topic_inst,
                constraint_inst,
                mode,
            ]
        else:
            MAPPING[int(row['index'])] = [
                inst_type,
                topic_inst + ' ' + constraint_inst,
                mode,
            ]


def get_instruction(datapoint, args, version):
    if args.data == 'wordnet':
        topic = datapoint['topic']
        constraint = datapoint['constraint']
        insert_position, *instruction_templates, mode = MAPPING[version]
        if insert_position == 'begin+end':
            instructions = [
                instruction_templates[0].format(topic),
                instruction_templates[1].format(constraint),
            ]
        elif insert_position == 'begin':
            instructions = [instruction_templates[0].format(constraint, topic), ""]
        elif insert_position == 'end':
            instructions = ["", instruction_templates[0].format(constraint, topic)]
        else:
            raise ValueError(f'`{insert_position}` not recognized.')

    elif args.data == 'wikidata':
        topic = datapoint['topic']
        constraint = datapoint['constraint']
        insert_position, *instruction_templates, mode = MAPPING[version]
        
        topic = get_wikidata_p_text(topic[0], topic[1])
        constraint = get_wikidata_p_text(constraint[0], constraint[1])
        
        if insert_position == 'begin+end':
            instructions = [
                instruction_templates[0].format(topic),
                instruction_templates[1].format(constraint),
            ]
        elif insert_position == 'begin':
            instructions = [instruction_templates[0].format(constraint, topic), ""]
        elif insert_position == 'end':
            instructions = ["", instruction_templates[0].format(constraint, topic)]
        else:
            raise ValueError(f'`{insert_position}` not recognized.')

    return dict(
        begin=instructions[0],
        end=instructions[1],
        insert_position=insert_position,
    )


def prepare_context(datapoint, args, version):
    new_datapoint = copy.deepcopy(datapoint)
    blocks = new_datapoint['context']
    blocks = ['== ' + sent.strip('==').strip() + ' ==' for sent in blocks.split('==\n')]
    
    instruction = get_instruction(new_datapoint, args, version)
    begin = instruction['begin']
    end = instruction['end']
    insert_position = instruction['insert_position']
    blocks = [begin] + blocks + [end]
    context_with_instructions = '\n'.join(blocks).strip()

    new_datapoint['context_with_instructions'] = context_with_instructions
    new_datapoint['version'] = version
    new_datapoint['begin'] = begin
    new_datapoint['end'] = end
    new_datapoint['insert_position'] = insert_position
    return new_datapoint
