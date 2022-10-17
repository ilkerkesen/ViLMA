# To-do:
# -Longformer
# -Electra
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from GRUEN.main import preprocess_candidates, get_grammaticality_score





def create_foils_from_lms(sentence, mask_target, declarative_statement, lm, grammer_threshold=0.8):

    """
    Create foils by using given Masked Language Modeling (MLM). Then apply filtering to created
    foils: 
        1) Do NLI (Roberta-large based) and if the output is "entailment (2)", 
           drop that sample. 
        2) Check grammer score by using GRUEN, if it is less than threshold, 
           drop that sample.
        3) 1 and 2 should be passed same time, AND operator.

    Args:
        sentence (string): Original sentence
        mask_target (string): Target to mask and create a foil for it
        declarative_statement (string): the declarative statement with the mask target
        lm (string): Language model that will be used for MLM.
        grammer_threshold(float): threshold for GRUEN grammer score
    Return:
        foils(list): list of dictionaries that stores the created foils, top5 prob included
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

    top_5_tokens      = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    top_5_probs       = torch.topk(mask_token_logits, 5, dim=1).values[0].tolist()

    foils = []
    for idx, token in enumerate(top_5_tokens):
        foiled_predicted_word = tokenizer.decode([token])
        foiled_declarative_statement = declarative_statement.replace(mask_target, foiled_predicted_word)
        sentence = sequence.replace(tokenizer.mask_token, foiled_predicted_word)
        probability = round(top_5_probs[idx], 3)
        nli_output = do_nli([declarative_statement, foiled_declarative_statement])
        grammetical_score = get_grammaticality_score(preprocess_candidates([foiled_declarative_statement]))
        if ((nli_output != 2) and grammetical_score[0] > grammer_threshold):
            sample = {"declarative statement": declarative_statement,
                    "foiled predicted word": foiled_predicted_word, 
                    "foiled declarative statement":foiled_declarative_statement, 
                    "sentence":sentence, 
                    "score": probability,
                    "nli_gruen_test": "pass"
                    }
        else:
            sample = {"declarative statement": declarative_statement,
                    "foiled predicted word": foiled_predicted_word, 
                    "foiled declarative statement":foiled_declarative_statement, 
                    "sentence":sentence, 
                    "score": probability,
                    "nli_gruen_test": "fail"
                    }
        foils.append(sample)
    return foils

# taken from fairseq repo
def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False, pad_to_length=None, pad_to_multiple=1, pad_to_bsz=None,):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def do_nli(source):
    """
    Do NLI task by using roberta large.
     0: contradiction
     1: neutral
     2: entailment

    Args:
        source(list): list that stores the foil and reference
    Return:
        out(int): integer value of whether the given source is 
                  contradiction (0), neutral (1) or entailment (2).
    """

    declarative_statement, foiled_declarative_statement = source

    # Download RoBERTa already finetuned for MNLI
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    roberta.eval() # disable dropout for evaluation

    # Encode a pair of sentences and make a prediction
    tokens = roberta.encode(declarative_statement, foiled_declarative_statement)
    logprobs = roberta.predict('mnli', tokens)
    out = logprobs.argmax(dim=1)

    return out.item()

    

    
