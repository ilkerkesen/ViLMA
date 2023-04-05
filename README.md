# vl-bench

This repository contains the following details about our V&L benchmark,

1. How to setup the data resources (e.g. video directories).
2. How to setup the models and run them.
3. How to evaluate the models.

## Data Resources

Our benchmark is built upon several different video resources. In this section, we share the details on how to setup these data resources.

### QUVA Repetition Dataset

This dataset is required to run the experiments for the repetitive action counting task. Use [this link](http://isis-data.science.uva.nl/tomrunia/QUVARepetitionDataset.tar.gz) to download the data. You can run the following command sequence to setup this data resource,

```bash
wget -c http://isis-data.science.uva.nl/tomrunia/QUVARepetitionDataset.tar.gz
tar -xvf QUVARepetitionDataset.tar.gz -C /path/to/extracted/
./bin/normalize_fps /path/to/quva/videos /path/to/quva/normalized_videos
python tasks/counting/normalize_annotations.py --data-dir /path/to/quva
```

This snippet first downloads the data, extracts it to somewhere, then performs FPS normalization on the data. For more details check [the counting documentation](/tasks/counting/README.md).

### Something-Something V2 Dataset

This dataset is required to run the experiments for the relation task. Follow the instruction reported to [this link](https://developer.qualcomm.com/software/ai-datasets/something-something) to download the data.

## Models

We share the details of each model in a seperate documentation file under `./models/{modality}/` directory. We have implemented the following models so far,

- [ClipBERT](./models/video/ClipBERT.md)
- [Frozen-in-Time](./models/video/FiT.md)
- [VideoCLIP](./models/video/VideoCLIP.md)
- [MCQ](./models/video/MCQ.md)
- [X-CLIP](./models/video/X-CLIP.md)

## Evaluation

Each adapted model produces a result annotation file which contains the predicted scores. We use a script to evaluate these result annotation files.

### Result Annotation File Format

Here is a dummy results annotation file,

```json
{
    "0": {
        "scores": [
            0.9998931884765625,
            0.9999251365661621
        ]
    },
    "1": {
        "scores": [
            0.9999557733535767,
            0.999962568283081
        ]
    }
}
```

These JSON files are actually key/value stores where the keys are the example ids (same with the data annotations). Each value is a `dict` which has the `scores` key. The `value` of this `scores` key represent the scores for each text input for the corresponding example. The first value *always* belongs to the true caption score. The remaining values are the scores for the foils in the same order as in the data annotation file. Note that, these scores could be either perplexity values, similarity scores or video-text matching probabilities, and this completely depends on the model.

### Running the Evaluation Script

Please do run the `./bin/eval.py` script to evaluate the models. It takes one argument which is the file path, and one option which specifies whether the model produces probabilities or scores (**TODO**: implement for the perplexity scores also as well). Here is an example,

```bash
python ./bin/eval.py /path/to/the/results/file.json --similarity
python ./bin/eval.py /path/to/the/results/file.json --probability
```

Passing `--probability` option makes the script treat the scores as probabilities, and allows user to produce the scores for the accuracy, precision and AUROC metrics. 
