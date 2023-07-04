import os.path as osp
from .utils import process_path


MISSING_VERBS = {
    'deseed': ('deseeding', 'deseeded', 'deseeds'),
    'microwave': ('microwaving', 'microwaved', 'microwaves'),
    'use': ('using', 'used', 'uses'),
    'wrap': ('wrapping', 'wrapped', 'wraps'),
    'endorse': ('endorsing', 'endorsed', 'enderoses'),
    'answering': ('answering', 'answered', 'answers'),
    'download': ('downloading', 'downloaded', 'downloads'),
    'unlocked': ('unlocking', 'unlocked', 'unlocks'),
    'braid': ('braiding', 'braided', 'braids'),
    'answers': ('answering', 'answered', 'answers'),
    'answered': ('answering', 'answered', 'answers'),
    'locked': ('locking', 'locked', 'locks'),
}


PLURAL_EXCEPTIONS = (
    'clothe',
    'corn',
    'cucumber',
    'flower',
    'pepper',
    'spoon',
)


# drill into
# wash -> wash some ...
# drink eggs
PLURALS = (
    'carrots',
    'clothes',
    'corn',
    'cucumbers',
    'eggs',
    'flowers',
    'hair',
    'peppers',  # FIXME: check it again.
    'shoes',
    'spoons',
)


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
PRESENT_INDEX = 3
PRESENT_CONT_INDEX = 5
PAST_PERFECT_INDEX = 11


def _get_present_tense(verb):
    if verb.endswith('sh') or verb.endswith('ch'):
        return f'{verb}es'
    elif verb.endswith('y'):
        return f'{verb[:-1]}ies'
    return f'{verb}s'


def get_present_tense(verb, verb_forms_dict=VERB_FORMS):
    forms = verb_forms_dict.get(verb, MISSING_VERBS.get(verb))
    if forms is None:
        return _get_present_tense(verb)
    if isinstance(forms, list) and forms[PRESENT_INDEX] == '':
        return None  # FIXME

    if isinstance(forms, list):
        return forms[PRESENT_INDEX]
    elif isinstance(forms, tuple):
        return forms[2]



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
    verb_ = verb.split()[0]
    verb_ = get_present_continuous_tense(verb_)
    verb_ = f'{verb_} ' + ' '.join(verb.split()[1:])
    verb_ = verb_.strip()
    noun_ = noun

    if verb == 'drill':
        verb_ = 'drilling into'

    is_plural = False
    if noun in PLURAL_EXCEPTIONS or verb == 'wash':  # FIXME: mostly.
        noun_ = f'{noun_}s'
        is_plural = True

    article = 'a'
    if noun_ in PLURALS or is_plural:
        article = 'some'
    elif noun_[0] in ('a', 'e', 'o', 'i'):
        article = 'an'
    else:
        article = 'a'
    return f'{verb_} {article} {noun_}'


def make_active_voice_sentence_with_noun_phrase(verb, noun_phrase):
    verb_ = verb.split()[0]
    verb_ = get_present_continuous_tense(verb_)
    verb_ = f'{verb_} ' + ' '.join(verb.split()[1:])
    verb_ = verb_.strip()
    if verb == 'drill':
        verb_ = 'drilling into'

    return f'{verb_} {noun_phrase}'


def make_foil_noun_phrases(nouns, verb):
    items = list()
    for noun in nouns:
        noun_ = noun
        is_plural = False
        if noun in PLURAL_EXCEPTIONS or verb == 'wash':  # FIXME: mostly.
            noun_ = f'{noun_}s'
            is_plural = True

        article = 'a'
        if noun_ in PLURALS or is_plural:
            article = 'some'
        elif noun_[0] in ('a', 'e', 'o', 'i'):
            article = 'an'
        else:
            article = 'a'
        noun_phrase = f'{article} {noun_}'

        item = {
            'noun': noun,
            'phrase': noun_phrase,
            'score': None,
        }
        items.append(item)
    return items

def make_proficiency_sentence(noun):
    noun_ = noun
    is_plural = noun_ in PLURALS
    if noun in PLURAL_EXCEPTIONS:
        noun_ = f'{noun_}s'
        is_plural = True

    article = 'a'
    if noun_ in PLURALS or is_plural:
        article = 'some'
    elif noun_[0] in ('a', 'e', 'o', 'i'):
        article = 'an'
    else:
        article = 'a'

    be = 'is' if not is_plural else 'are'
    return f'there {be} {article} {noun_}'


def make_passive_voice_sentence(verb, noun):
    verb_ = get_past_perfect_tense(verb)
    if verb == 'drill':
        verb_ = 'drilling into'
    return f'the {noun} is being {verb_}'