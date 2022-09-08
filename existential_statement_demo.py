import sys
import random
import copy
import spacy
from spacy import displacy
#from GRUEN.main import get_grammaticality
from transformers import AutoModel

model = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
#sys.exit()
# below we have a list of lists. the innermost list has elements that could replace any other element within the same list
# to create a foil that is guaranteed to be valid.
# e.g.: replacing "person" by "animal" (or vice-versa) will always result in a valid foil.
foil_nouns = [
    ["dog", "cat", "elephant", "giraffe", "hamster", "duck"],
    ["dogs", "cats", "elephants", "giraffes", "hamsters", "ducks"],
    ["person", "animal"],
    ["people", "animals"],
    ["man", "woman"],
    ["men", "women"],
    ["boy", "girl", "adult"],
    ["boys", "girls", "adults"],
    ["park", "street", "house", "pier"],
    ["crumbs", "fruits", "glasses"]
]

def create_foils_from_nouns(sentence):
    foil_sentences = []
    for foil_set in foil_nouns:
        for word in sentence.split(" "):
            if word in foil_set:
                foil_set_ = copy.copy( foil_set )
                foil_set_.remove( word )
                # choose a random word from the set to substitute
                word_replacement = random.choice( foil_set_ )
                a_to_an = False
                if word[0] in ["a", "e", "i", "o", "u"] and word_replacement[0] not in ["a", "e", "i", "o", "u"]:
                    a_to_an = True

                an_to_a = False
                if word[0] not in ["a", "e", "i", "o", "u"] and word_replacement[0] in ["a", "e", "i", "o", "u"]:
                    an_to_a = True

                if an_to_a and "a " + word in sentence:
                    s = sentence.replace("a " + str(word), "an " + str(word_replacement))
                elif a_to_an and "an " + word in sentence:
                    s = sentence.replace("an " + str(word), "a " + str(word_replacement))
                else:
                    s = sentence.replace(str(word), str(word_replacement))
                foil_sentences.append( s )
    return foil_sentences

example_captions = [
        "A boy running in a park with a dog",
        "Two ducks are fighting for some bread crumbs",
        "In this image there is a person",
        "There are some dogs running in the park"
]

#get_grammaticality( example_captions )

# Load the language model
nlp = spacy.load("en_core_web_trf")

#sentence = 'Deemed universities charge huge fees'
all_partial_statements = {}

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
        print ("{:<15} | {:<8} | {:<8} | {:<15} | {:<20}"
             .format(str(token.text), str(token.lemma_), str(token.dep_), str(token.head.text), str([child for child in token.children])))
        ## Use displayCy to visualize the dependency 
        #displacy.render(doc, style='dep', jupyter=False, options={'distance': 120})

        if not there_tobe_construction_found:
            if token.dep_ == "nsubj" or token.dep_ == "pobj":
                # singular form
                if token.text == token.lemma_:
                    det = "a" if token.text[0] not in ["a","e","i","o","u"] else "an"
                    partial_declarative_statements.append("There is " + det + " " + token.text)
                else:
                    partial_declarative_statements.append("There are some " + token.text)
        else:
            if token.dep_ == "attr" or token.dep_ == "pobj":
                # singular form
                if token.text == token.lemma_:
                    det = "a" if token.text[0] not in ["a","e","i","o","u"] else "an"
                    partial_declarative_statements.append("There is " + det + " " + token.text)
                else:
                    partial_declarative_statements.append("There are some " + token.text)

    #all_partial_statements.extend( partial_declarative_statements )
    all_partial_statements[ idx ] = {"original_sentence":sentence, "correct":[], "foiled":[]}
    #all_partial_statements[idx]["original_sentence"] = sentence

    print()
    print("Partial statements:")
    print(partial_declarative_statements)
    for ds in partial_declarative_statements:
        foil_sentences = create_foils_from_nouns( ds )
        #all_partial_statements.extend( foil_sentences )
        print("Foils: ")
        print(foil_sentences)
        all_partial_statements[idx]["correct"].append( ds )
        all_partial_statements[idx]["foiled"].append( foil_sentences )


    print("All partial statements (correct and foils):")
    print(all_partial_statements)
