import json
from os.path import expanduser
from argparse import ArgumentParser

from lemminflect import getInflection
from intransitive_verbs import INTRANSITIVE


PTB_TAG = {
    "VB": "base form",
    "VBD": "past tense",
    "VBG": "gerund or presnet participle",
    "VBN": "past participle",
    "VBP": "non-3rd person singular present",
    "VBZ": "3rd person singular present",
}


def inflect(verb, tag):
    assert tag in PTB_TAG, f"pos '{tag}' not in {list(tag.keys())}"
    # simple manager of multi-word verbs - maybe append the second part of the verb?
    particle = None
    if " " in verb:
        verb, particle = verb.split(" ")
    infl = getInflection(verb, tag=tag)
    if len(infl) == 0:
        print(f"- Error inflecting '{verb}' with tag '{tag}'")
        infl = ["[UNK]"]
    return infl[0], particle


def not_transitive(verb):
    if verb in INTRANSITIVE:
        return True
    else:
        return False


def is_plural(subject):
    if subject[-1] == "s" or "and" in subject:
        if subject[-2] == "s":  # e.g., "glass"
            return False
        return True
    elif " and " in subject:
        return True
    else:
        return False


def manage_aux(subject):
    if subject is None:
        raise ValueError("Subject cannot be None")
    if is_plural(subject):
        return "are"
    else:
        return "is"


def create_foils(
    sentence, reduced, target, foil_types, start_adverb, central_adverb, end_adverb
):
    foils = {}
    switcher = {
        "action": get_action_capt_and_foil,
        "pre-state": get_prestate_capt_and_foil,
        "post-state": get_poststate_capt_and_foil,
        "inverse": get_inverse_capt_and_foil,
    }
    for ft in foil_types:
        foils[ft] = switcher[ft](
            sentence=sentence,
            target=target,
            reduced=reduced,
            start_adverb=start_adverb,
            central_adverb=central_adverb,
            end_adverb=end_adverb,
        )
    return {"foils": foils}


def get_action_capt_and_foil(
    sentence,
    target,
    reduced,
    start_adverb="Initially",
    central_adverb="Then",
    end_adverb="At the end",
):
    _object = sentence["object"] if sentence["object"] else "something"
    tag = "VBZ" if not is_plural(_object) else "VBP"
    _verb, _particle = inflect(sentence[target], tag=tag)
    _verb_passive, _particle_passive = inflect(sentence[target], tag="VBZ")
    _inverse, _particle_inv = inflect(sentence["state-inverse"], tag=tag)
    _inverse_passive, _particle_inv_passive = inflect(
        sentence["state-inverse"], tag="VBZ"
    )
    aux = manage_aux(_object)

    if not_transitive(sentence[target]):
        if reduced:
            capt = (
                f"{_object.capitalize()} {_verb}{' ' + _particle if _particle else ''}."
            )
            foil = f"{_object.capitalize()} {_inverse}{' ' + _particle_inv if _particle_inv else ''}."
        else:
            # EXTENDED VERSION
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, {_object} {_verb}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, {_object} {_inverse}{' ' + _particle_inv if _particle_inv else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
    else:
        if reduced:
            capt = f"Someone {_verb_passive} {_object}{' ' + _particle_passive if _particle_passive else ''}."
            foil = f"Someone {_inverse_passive} {_object}{' ' +_particle_inv_passive if _particle_inv_passive else ''}."
        else:
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, someone {_verb_passive} {_object}{' ' + _particle_passive if _particle_passive else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['pre-state']} {central_adverb}, someone {_inverse_passive} {_object}{' ' + _particle_inv_passive if _particle_inv_passive else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
    return capt, foil


def get_prestate_capt_and_foil(
    sentence,
    target,
    reduced,
    start_adverb="Initially",
    central_adverb="Then",
    end_adverb="At the end",
):
    _object = sentence["object"] if sentence["object"] else "Someone"
    aux = manage_aux(_object)
    _verb, _particle = inflect(sentence[target], tag="VBZ")
    aux = manage_aux(_object)

    if not_transitive(sentence[target]):
        if reduced:
            # REDUCED VERISON
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['post-state']}."

        else:
            # EXTENDED VERSION
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, {_object} {_verb}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['post-state']}. {central_adverb}, {_object} {_verb}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
    else:
        if reduced:
            # REDUCED VERSION
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['post-state']}."
        else:
            # EXTENDED VERSION
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, someone {_verb} {_object}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['post-state']} {central_adverb}, someone {_verb} {_object}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
    return capt, foil


def get_poststate_capt_and_foil(
    sentence,
    target,
    reduced,
    start_adverb="Initially",
    central_adverb="Then",
    end_adverb="At the end",
):
    _object = sentence["object"] if sentence["object"] else "Someone"
    _verb, _particle = inflect(sentence[target], tag="VBZ")
    aux = manage_aux(_object)

    if not_transitive(sentence[target]):
        if reduced:
            capt = f"At the end, {_object} {aux} {sentence['post-state']}."
            foil = f"At the end, {_object} {aux} {sentence['pre-state']}."
        else:
            # EXTENDED VERSION
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, {_object} {_verb}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, {_object} {_verb}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['pre-state']}."
    else:
        if reduced:
            capt = f"At the end, {_object} {aux} {sentence['post-state']}."
            foil = f"At the end, {_object} {aux} {sentence['pre-state']}."

        # EXTENDED VERSION
        else:
            capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, someone {_verb} {_object}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
            foil = f"{start_adverb}, {_object} {aux} {sentence['pre-state']} {central_adverb}, someone {_verb} {_object}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['pre-state']}."
    return capt, foil


def get_inverse_capt_and_foil(
    sentence,
    target,
    reduced=None,
    start_adverb="Initially",
    central_adverb="Then",
    end_adverb="At the end",
):
    _verb, _particle = inflect(sentence[target], tag="VBZ")
    _inverse, _particle_inv = inflect(sentence["state-inverse"], tag="VBZ")
    _object = sentence["object"] if sentence["object"] else "something"
    aux = manage_aux(_object)

    if not_transitive(sentence[target]):
        capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, {_object} {_verb}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
        foil = f"{start_adverb}, {_object} {aux} {sentence['post-state']}. {central_adverb}, {_object} {_inverse}{' ' + _particle_inv if _particle_inv else ''}. {end_adverb}, {_object} {aux} {sentence['pre-state']}."
    else:
        capt = f"{start_adverb}, {_object} {aux} {sentence['pre-state']}. {central_adverb}, someone {_verb} {_object}{' ' + _particle if _particle else ''}. {end_adverb}, {_object} {aux} {sentence['post-state']}."
        foil = f"{start_adverb}, {_object} {aux} {sentence['post-state']}. {central_adverb}, someone {_inverse} {_object}{' ' + _particle_inv if _particle_inv else ''}. {end_adverb}, {_object} {aux} {sentence['pre-state']}."
    return capt, foil


def main(args):
    datapath = expanduser(args.data)
    # outpath_dir = "data/"
    foil_types = ["action", "pre-state", "post-state", "inverse"]
    reduced = not (args.extended)
    start_adverb = args.start_adverb.capitalize()
    central_adverb = args.central_adverb.capitalize()
    end_adverb = args.end_adverb.capitalize()
    target = "verb-hypernym" if not args.preserve_original else "verb"

    data = json.load(open(datapath))
    print(f"- foiling data at {datapath} - {len(data)} examples")

    datasets = [{} for f in foil_types]
    for k, v in data.items():
        _foils = create_foils(
            v["change_of_state"],
            foil_types=foil_types,
            target=target,
            reduced=reduced,
            start_adverb=start_adverb,
            central_adverb=central_adverb,
            end_adverb=end_adverb,
        )
        for i, _v in enumerate(_foils["foils"].values()):
            datasets[i][k] = v.copy()
            capt = _v[0]
            foil = _v[1]
            datasets[i][k].update({"caption": capt, "foils": [foil]})

    for i, ftype in enumerate(foil_types):
        json.dump(datasets[i], open(f"data/change-state-{ftype}.json", "w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Create extended captions and foils, containing all of the change-of-state sub-phases (pre-state, action, post-state)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/change-state-base.json",
        help="Path to the extracted and parsed original sentences",
    )
    parser.add_argument(
        "--start-adverb",
        type=str,
        default="Initially",
        help="The adverb to use in the template to denote the pre-state sub-phase",
    )
    parser.add_argument(
        "--central-adverb",
        type=str,
        default="Then",
        help="The adverb to use in the template to denote the action sub-phase",
    )
    parser.add_argument(
        "--end-adverb",
        type=str,
        default="At the end",
        help="The adverb to use in the template to denote the post-state sub-phase",
    )
    parser.add_argument(
        "--preserve-original",
        action="store_true",
        help="In the true caption, preserve the original verb as extracted from the original sentence (otherwise, replace it with its hypernym)",
    )
    args = parser.parse_args()
    main(args)
