import spacy, sys, time, json
from utils import create_foils_from_lms


#LM = sys.argv[1]
LM = "albert"

# supported lms
models = ["bert", "albert", "spanbert", "roberta", "bart", "t5"]
if LM not in models:
    raise Exception("Language model is not supported, please change it!")

example_captions = ["A boy running in a park with a dog",
                    "Two ducks are fighting for some bread crumbs",
                    "In this image there is a person",
                    "There are some dogs running in the park"]

# Load the language model
nlp = spacy.load("en_core_web_trf")

#sentence = 'Deemed universities charge huge fees'
all_foils={}
for _idx, sentence in enumerate(example_captions):
    # we are generating a dummy idx to uniquely identify each example
    idx = _idx

    partial_declarative_statements = []
    print(sentence)
    # nlp function returns an object with individual token information, 
    # linguistic features and relationships
    doc = nlp(sentence)

    print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}".format('Token', 'Lemma', 'Relation','Head', 'Children'))
    print ("-" * 70)

    # Find whether there are constructions "There is" or "There are":
    previous_token = None
    there_tobe_construction_found = False
    for token in doc:
        if token.dep_ == "ROOT" and (token.text == "is" or token.text == "are"):
            if not previous_token is None and previous_token.text.lower() == "there":
                #print("YES: ", doc)
                there_tobe_construction_found = True
        previous_token = token

    for token in doc:
        # Print the token, dependency nature, head and all dependents of the token
        print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}".format(str(token.text), str(token.lemma_), str(token.dep_), str(token.head.text), str([child for child in token.children])))

        if not there_tobe_construction_found:
            if token.dep_ == "nsubj" or token.dep_ == "pobj":
                # singular form
                if token.text == token.lemma_:
                    det = "a" if token.text[0] not in ["a","e","i","o","u"] else "an"
                    mask_target = token.text
                    partial_declarative_statements.append((sentence, mask_target, "There is " + det + " " + token.text))
                else:
                    mask_target = token.text
                    partial_declarative_statements.append((sentence, mask_target, "There are some " + token.text))
        else:
            if token.dep_ == "attr" or token.dep_ == "pobj":
                # singular form
                if token.text == token.lemma_:
                    det = "a" if token.text[0] not in ["a","e","i","o","u"] else "an"
                    mask_target = token.text
                    partial_declarative_statements.append((sentence, mask_target, "There is " + det + " " + token.text))
                else:
                    mask_target = token.text
                    partial_declarative_statements.append((sentence, mask_target, "There are some " + token.text))

    # create foils
    foils={}
    for part_idx, ds in enumerate(partial_declarative_statements):
        sentence, mask_target, declarative_statement = ds
        foil = create_foils_from_lms(sentence, mask_target, declarative_statement, LM)
        foils["part"+str(part_idx)] = foil

    all_foils[idx] = foils

with open("foils.json", "w") as fp:
    json.dump(all_foils, fp, indent=4)


    
