import spacy, sys, time, json, argparse
from utils import create_action_foils_from_lms
start_time = time.time()

# args
parser = argparse.ArgumentParser()
parser.add_argument("--lm",              required=True, help="Choose: 'bert', 'albert', 'spanbert', 'roberta', 'bart', 't5'")
parser.add_argument("--profiency_task",  required=True, help="foiling either 'actions' or 'noun'")
parser.add_argument("--data",            required=True, help="andrea, ilker, michele, mustafa")
parser.add_argument("--gruen_threshold", default=0.8,   help="GRUEN Score hyperparameter")
args = parser.parse_args()
models = ["bert", "albert", "spanbert", "roberta", "bart", "t5"]
if args.lm not in models:
    raise Exception("Language model is not supported, please change it!")

if args.data == "andrea":
    with open("/kuacc/users/eacikgoz17/vl-bench/vl-bench/data/change-state.json") as f:
        data = json.load(f)
        captions = [i["original_caption"] for i in data.values()]
        captions_long = [i["caption"] for i in data.values()]
elif args.data == "ilker":
    with open("vl-bench/data/dummy.json") as f:
        data = json.load(f)
        breakpoint()
elif args.data == "michele":
    with open("/kuacc/users/eacikgoz17/vl-bench/vl-bench/data/relations.json") as f:
        data = json.load(f)
        captions = [i["caption"] for i in data.values()]
elif args.data == "mustafa":
    with open("/kuacc/users/eacikgoz17/vl-bench/vl-bench/data/mustafa.json") as f:
        data = json.load(f)
        captions = [i["corrected_sentence"] for i in data]

"""# dataset
example_captions = [
    "someone is hammering a banana.",
    "someone cuts a keyboard.",
]"""

# Load the language model
nlp = spacy.load("en_core_web_trf")

all_foils={}
for _idx, sentence in enumerate(captions[20:]):
    # we are generating a dummy idx to uniquely identify each example
    idx = _idx

    data = []
    print(sentence)
    # nlp function returns an object with individual token information, linguistic features and relationships
    doc = nlp(sentence)

    print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}".format('Token', 'Lemma', 'Relation','Head', 'Children'))
    print ("-" * 70)

    tokens_dependency = [token.dep_ for token in doc]
    for token in doc:
        # print the token, dependency nature, head and all dependents of the token
        print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}".format(str(token.text), str(token.lemma_), str(token.dep_), str(token.head.text), str([child for child in token.children])))

        if args.profiency_task == "action":
            if token.dep_ == "ROOT":
                mask_target = token.text
                data.append((sentence, mask_target))
        elif args.profiency_task == "noun":
            if ("dobj" in tokens_dependency) and ("pobj" in tokens_dependency): # both in the caption mask pobj
                if token.dep_ == "pobj":
                    mask_target = token.text
                    print(f"Mask Target: {mask_target}")
                    data.append((sentence, mask_target))
            elif ("dobj" in tokens_dependency) and ("pobj" not in tokens_dependency): # dobj in the caption mask dobj
                if token.dep_ == "dobj":
                    mask_target = token.text
                    print(f"Mask Target: {mask_target}")
                    data.append((sentence, mask_target))
            elif ("dobj" not in tokens_dependency) and ("pobj" in tokens_dependency): # pobj in the caption mask pobj
                if token.dep_ == "pobj":
                    mask_target = token.text
                    print(f"Mask Target: {mask_target}")
                    data.append((sentence, mask_target))

    # create foils
    foils={}
    for part_idx, d in enumerate(data):
        sentence, mask_target = d
        foil = create_action_foils_from_lms(sentence, mask_target, args.lm, args.gruen_threshold)
        foils["part"+str(part_idx)] = foil
    all_foils[idx] = foils

# write created foils to json file
with open(args.data + "_" + args.profiency_task + "_foils_"+ args.lm + ".json", "w") as fp:
    json.dump(all_foils, fp, indent=4)
end_time = time.time()
print(f"Script run time: {end_time - start_time}")
    
