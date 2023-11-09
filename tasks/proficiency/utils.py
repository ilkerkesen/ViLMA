import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from GRUEN.main import preprocess_candidates, get_grammaticality_score

def create_proficiency_foils_from_lms(sentence, mask_target, lm, proficiency_task, grammer_threshold=0.8):
    """
    Create action foils by using given Masked Language Modeling (MLM). Then apply filtering to created
    foils: 
        step1 => Do NLI (Roberta-large based) and if the output is "entailment (2)", 
                 drop that sample. 
        step2 => Check grammer score by using GRUEN, if it is less than threshold, 
                 drop that sample.
        step3 => 1 and 2 should be passed same time, AND operator.

    Args:
        dataset (dict): loaded data
        sentence (string): Original sentence
        mask_target (string): Target to mask and create a foil for it
        lm (string): Language model that will be used for MLM.
        grammer_threshold(float): threshold for GRUEN grammer score
    Return:
        foils (list): list of dictionaries that stores the created foils, top5 prob included
                      for one foil.
    """

    mask_before, mask_after = sentence.split(mask_target)

    if lm == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model     = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    elif lm == "albert":
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        model     = AutoModelForMaskedLM.from_pretrained("albert-base-v2")
    elif lm == "spanbert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model     = AutoModelForMaskedLM.from_pretrained("SpanBERT/spanbert-base-cased")
    elif lm == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model     = AutoModelForMaskedLM.from_pretrained("roberta-base")
    elif lm == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        model     = AutoModelForMaskedLM.from_pretrained("facebook/bart-large")
    elif lm == "t5":
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model     = AutoModelForMaskedLM.from_pretrained("t5-base")

    sequence          = (f"{mask_before}{tokenizer.mask_token}{mask_after}.")
    inputs            = tokenizer(sequence, return_tensors="pt")
    mask_token_index  = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    token_logits      = model(**inputs)[0]
    #token_logits      = model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens      = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()
    top_5_probs       = torch.topk(mask_token_logits, 3, dim=1).values[0].tolist()

    foils = {}
    nli_dict = {0: "contradiction", 1: "neutral", 2: "entailment"}
    for idx, token in enumerate(top_5_tokens):
        foiled_predicted_word = tokenizer.decode([token])
        foiled_sentence = sequence.replace(tokenizer.mask_token, foiled_predicted_word)
        probability = round(top_5_probs[idx], 3)
        nli_output = do_nli([sentence, foiled_sentence])
        grammetical_score = get_grammaticality_score(preprocess_candidates([foiled_sentence]))
        if ((nli_output != 2) and (grammetical_score[0] > grammer_threshold)):
            nli_gruen_test = "pass"
        else:
            nli_gruen_test = "fail"
        sample = {"proficiency_test": proficiency_task,
                  "foiled_sentence":foiled_sentence,
                  "mask_target": mask_target,
                  "foiled_predicted_word": foiled_predicted_word, 
                  "lm": lm,
                  "lm_score": probability,
                  "nli_prediction": nli_dict[nli_output],
                  "gruen_score": round(grammetical_score[0], 5),
                  "nli_gruen_test": nli_gruen_test,
                 }  
        foils[str(idx)] = sample
    return foils

def do_nli(source):
    """
    Do NLI task by using roberta large.
     0: contradiction
     1: neutral
     2: entailment

    Args:
        source (list): list that stores the foil and reference
    Return:
        out (int): integer value of whether the given source is 
                   contradiction (0), neutral (1) or entailment (2).
    """

    declarative_statement, foiled_declarative_statement = source

    # Download RoBERTa already finetuned for MNLI
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    roberta.eval() 

    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode(declarative_statement, foiled_declarative_statement)
    logprobs = roberta.predict('mnli', tokens)
    out = logprobs.argmax(dim=1)

    return out.item()

    

    
