# Rare Actions -- Unusual Interactions

This part of the repository contains the complete pipeline to generate the rare actions task in the proposed benchmark.
The complete process is presented step-by-step as follows,

1. Get the original RareAct annotation file `rareact.csv`.
2. Download the videos.
3. Detect objects in the downloaded videos.
4. Create the foil candidates.
5. Filter out the low quality candidates manually.
6. Create the annotations.

## Detect Objects
Use the `create_captions.py` script,

```bash
python tasks/actions/detect_objects.py \ 
    -i /path/to/rareact.csv \
    -o $DATA_DIR/detected-objects.json \
    --video-dir /path/to/videos
```

## Generate Foil Candidates for the Action Replacement
We use T5 model to generate foil candidates for the action replacement subtask, because T5 works better for predicting actions, i.e. some verbs consists of multiple words (e.g. typing on, drilling into etc.).
```bash
python t5.py \ 
    -i /path/to/rareact.csv \
    -o $DATA_DIR/action-candidates-t5-large.json \
    --top-p 0.5 --do-sample
```

This command will generate the candidates for the action replacement task. I manually filtered them in order to generate better foils. Note that I also filtered out some actions commonly used (e.g. be, have, see, use) or actions that indicate only making contact (e.g. pick, reach, touch). The model also tends to generate some adverbs and I filtered out them too.

## Generate Foil Candidatea for the Object Replacement
We now switch to masked language models in this part, since we observed that span generation with T5 for the objects did not work out well. We mask the noun/object, then generate foil candidates using several different MLMs using three different determiners `a`, `an` and `some`. We specifically used `bert-large-uncased`, `albert-large-v2` and `roberta-large`,

```bash
python mlm.py \
    -i /path/to/rareact.csv \
    -o $DATA_DIR/object-candidates-$MODEL_NAME-top-1024.json \
    --model-name $MODEL_NAME \
    -T 0.01
```

We then ensemble these outputs,

```bash
python ensemble_mlm_outputs.py \
    -i $DATA_DIR/object-candidates-roberta-large-top-1024.json \
    -i $DATA_DIR/object-candidates-albert-large-v2-top-1024.json \
    -i $DATA_DIR/object-candidates-bert-large-uncased-top-1024.json \
    -o $DATA_DIR/object-candidates-ensembled.json
```

After that we again filter out the implausible candidates manually.

## Creating the subtasks
We simply run the following commands,

```bash
python create_action_subset.py \
    -i $DATA_DIR/action-candidates.json \
    -d /path/to/rareact.csv \
    -o ../../data/rare-actions-verb-foils.json \
    --video-dir /path/to/video-dir \
    --object-detection-file $DATA_DIR/detected-objects.json \
    --num-examples 1000

python create_object_subset.py \
    -i $DATA_DIR/object-candidates.json \
    -d /path/to/rareact.csv \
    -o ../../data/rare-actions-noun-foils.json \
    --video-dir /path/to/video-dir \
    --object-detection-file $DATA_DIR/detected-objects.json \
    --num-examples 1000
```