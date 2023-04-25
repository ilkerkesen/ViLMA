# Rare Actions -- Unusual Interactions

This part of the repository contains the complete pipeline to generate the rare actions task in the proposed benchmark.
The complete process is presented step-by-step as follows,

1. Get the original RareAct annotation file `rareact.csv`.
2. Create the captions by running `create_captions.py`.
3. Create the foils by running `create_foils.py`
4. Compute the nli scores by running `compute_nli.py`.
5. Filter out the entailment cases by using `filter_entailments.py`.
6. Run the object detector on the videos by using `detect_objects.py`.
7. Filter out the objects by using `filter_objects.py`.
8. Merge the two annotations file: the object foils / the action foils.

## Creating Captions
Use the `create_captions.py` script,

```bash
python create_captions.py -i $DATA_DIR/rareact.csv -o $DATA_DIR/active-only-captions.json --active
python create_captions.py -i $DATA_DIR/rareact.csv -o $DATA_DIR/passive-only-captions.json --passive
```

## Foil Generation
Use the `create_foils.py` script,
```bash
python create_foils.py -i $DATA_DIR/active-only-captions.json -o $DATA_DIR/active-foil-verb.json --foil-verb
python create_foils.py -i $DATA_DIR/passive-only-captions.json -o $DATA_DIR/passive-foil-verb.json --foil-verb
python create_foils.py -i $DATA_DIR/active-only-captions.json -o $DATA_DIR/active-foil-noun.json --foil-noun
python create_foils.py -i $DATA_DIR/passive-only-captions.json -o $DATA_DIR/passive-foil-noun.json --foil-noun
```

## NLI Filtering
First, generate produce the NLI scores and predict the NLI classes,

```bash
python compute_nli.py -i $DATA_DIR/active-foil-verb.json -o $DATA_DIR/active-foil-verb-nli-scores.json
python compute_nli.py -i $DATA_DIR/active-foil-noun.json -o $DATA_DIR/active-foil-noun-nli-scores.json
python compute_nli.py -i $DATA_DIR/passive-foil-verb.json -o $DATA_DIR/passive-foil-verb-nli-scores.json
python compute_nli.py -i $DATA_DIR/passive-foil-noun.json -o $DATA_DIR/passive-foil-noun-nli-scores.json
```

After you get the NLI predictions, filter out the entailments,

```bash
python filter_entailments.py -i $DATA_DIR/active-foil-verb-nli-scores.json -o $DATA_DIR/active-foil-verb-nli-filtered.json
python filter_entailments.py -i $DATA_DIR/active-foil-noun-nli-scores.json -o $DATA_DIR/active-foil-noun-nli-filtered.json
python filter_entailments.py -i $DATA_DIR/passive-foil-verb-nli-scores.json -o $DATA_DIR/passive-foil-verb-nli-filtered.json
python filter_entailments.py -i $DATA_DIR/passive-foil-noun-nli-scores.json -o $DATA_DIR/passive-foil-noun-nli-filtered.json
```

## Object Filtering
Again, first run the detector,

```bash
python detect_objects.py -i $DATA_DIR/active-foil-verb-nli-filtered.json -o $DATA_DIR/detected_objects.json
```

In this stage, the input file does not matter: it could be any previous file since we only need the video ids and start/end timestamps. Then, we filter out the detected objects in the foils,

```bash
python filter_objects.py -a $DATA_DIR/active-foil-noun-nli-filtered.json -d $DATA_DIR/detected_objects.json -o $DATA_DIR/active-foil-noun-filtered.json
python filter_objects.py -a $DATA_DIR/passive-foil-noun-nli-filtered.json -d $DATA_DIR/detected_objects.json -o $DATA_DIR/passive-foil-noun-filtered.json
```

## Generating the Final Annotations File
```bash
python create_uniform_annotations.py -v $DATA_DIR/active-foil-verb-nli-filtered.json -n $DATA_DIR/active-foil-noun-filtered.json -o $DATA_DIR/active-uniform.json
python create_uniform_annotations.py -v $DATA_DIR/passive-foil-verb-nli-filtered.json -n $DATA_DIR/passive-foil-noun-filtered.json -o $DATA_DIR/passive-uniform.json
```