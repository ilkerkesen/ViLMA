import os.path as osp
from .utils import process_path


MISSING_VERBS = {
    'deseed': ('deseeding', 'deseeded'),
    'microwave': ('microwaving', 'microwaved'),
    'use': ('using', 'used'),
    'wrap': ('wrapping', 'wrapped'),
    'endorse': ('endorsing', 'endorsed'),
    'answering': ('answering', 'answered'),
    'download': ('downloading', 'downloaded'),
    'unlocked': ('unlocking', 'unlocked'),
    'braid': ('braiding', 'braided'),
    'answers': ('answering', 'answered'),
    'answered': ('answering', 'answered'),
    'locked': ('locking', 'locked'),
}


def create_verb_forms_dict(file_path):
    '''
        Processes the en-verbs.txt file. Listing the most common useful forms,
        - 0: the verb lemma (e.g. throw)
        - 3: the third person singular (e.g. throws)
        - 5: the present continous tense (e.g. throwing)
        - 10: the past tense (e.g. threw)
        - 11: the past perfect tense (e.g. thrown)
    '''
    file_path = process_path(file_path)
    verb_forms_dict = dict()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith(';;;'):
                forms = line.strip().split(',')
                verb_forms_dict[forms[0]] = forms
    return verb_forms_dict


VERB_FORMS_FILE = osp.join(osp.dirname(process_path(__file__)), 'en-verbs.txt')
VERB_FORMS = create_verb_forms_dict(VERB_FORMS_FILE)
PRESENT_CONT_INDEX = 5
PAST_PERFECT_INDEX = 11


def get_present_continuous_tense(verb, verb_forms_dict=VERB_FORMS):
    forms = verb_forms_dict.get(verb, MISSING_VERBS.get(verb))
    if forms is None:
        return f'{verb}ing'
    if isinstance(forms, list) and forms[PRESENT_CONT_INDEX] == '':
        raise RuntimeError(
            f"The present continuous tense doesn't exist for {verb}")

    if isinstance(forms, list):
        return forms[PRESENT_CONT_INDEX]
    elif isinstance(forms, tuple):
        return forms[0]


def get_past_perfect_tense(verb, verb_forms_dict=VERB_FORMS):
    forms = verb_forms_dict.get(verb, MISSING_VERBS.get(verb))
    if forms is None:
        return f'{verb}ed'
    if isinstance(forms, list) and forms[PAST_PERFECT_INDEX] == '':
        raise RuntimeError(
            f"The past perfect tense doesn't exist for {verb}.")
    
    if isinstance(forms, list):
        return forms[PAST_PERFECT_INDEX]
    elif isinstance(forms, tuple):
        return forms[1]


def make_active_voice_sentence(verb, noun):
    verb_ = get_present_continuous_tense(verb)
    return f'{verb_} the {noun}'


def make_passive_voice_sentence(verb, noun):
    verb_ = get_past_perfect_tense(verb)
    return f'the {noun} is being {verb_}'