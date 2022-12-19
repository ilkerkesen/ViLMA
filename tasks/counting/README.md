# Repetitive Action Counting

This document summarizes the work we have done for the repetitive action counting. This work consists of three pieces,

1. Collecting textual annotations for QUVA dataset.
2. Generating data annotations w/ foils for pretrained video-text models.
3. Adapting some models and performing preliminary experiments.

## Collecting Textual Annotations

### Initial Decisions

We first started with annotating data. Originally, QUVA doesn't include any type of textual annotations for the videos, so we manually created caption templates for each video. Each template includes __exactly \<number\>__ phrase. We included the word _exactly_ because we wanted to be sure about the models count action throughout the whole video. We also tried to mention specific details about the scene or the actors (e.g. _a man in a red t-shirt_). We mixed present continous tense (a man is doing exactly \<number\> push-ups.) and present tense (a man does exactly \<number\> push-ups.). We tried to collect multiple templates to have a variety. The first 75 templates are constructed by Ilker, and the remaining 25 templates constructed by Emre Can. To create captions and foils, we simply replace \<number\> with the correct number and a false number. We also annotated actions together with their categories. Here's an example JSON entry,

```json
"003": {
    "id": "003",
    "prefix": "003_table_tennis",
    "category": "exercise",
    "action": "table tennis",
    "templates": [
        "each table tennis player hits the ball exactly <number> times.",
        "each player hits the ball exactly <number> times using their rackets."
    ],
    "place": "gym"
}
```

The key value of this entry is the dataset index, which is same with the value of _id_ field. _prefix_ field makes it easy to read files to create the final JSON annotation files. We also tried to categorize and annotate the actions, which is exercise in the above example. We also annotated the place where the video happens, but we don't have any plans to use them right now (we did it because it was cheap in terms of time).

To construct textual templates I followed this process,

- Check the back-translation. I first created the textual template, then using Google Translate I checked whether the text changes after a few translation between some language. The other language is Turkish in my case, because I'm native, so I can agree or disagree about the translation.
- Checked YouTube / Google Images search results for the phrases. This is also similar to back-translation, but more visually grounded fashion.
- Skimmed some blog posts. There are videos which simply require knowledge of something (e.g. playing guitar / violin, or doing fitness exercise). So, in order to find the right words, I scanned some blog posts when I needed.

### Meeting with Albert

We also met with Albert on 25th October. We fixed syntax of the textual templates all together, and went over the semantics / the meanings of the constructed templates. While fixing the templates, we obeyed the rules of British English (e.g. skipping rope instead of jumping rope). We also switched to present tense by abandoning present continious tense, as suggested by Albert. We also replaced specific terminology with more general expressions (e.g. use lifting weights instead of doing skullcrushers). Lastly, we also discussed how to create


## Video FPS Normalization
Some models, i.e. faireq models, only work with a fixed FPS rate. However, the original QUVA dataset includes videos with varying FPS rates. To make the data compatible with the models, we simply perform a two-stage normalization: (i) we normalize the annotations (`./normalize_annotations.py`), (ii) we upsample the videos (`bin/normalize_fps`).


## Generating Annotations

This part describes how we automatically create the final JSON annotations consumed by the models using the templates. To achieve this, we implemented a script named `create_quva_annotations.py`. This script takes 4 required input options,

1. `--input-file`: Template file path.
2. `--output-file`: The path to save the generated annotations.
3. `--data-dir`: The QUVA dataset root path.
4. `--method`: The method that is going to be used to create annotations.

Among these options, `--method` is the most crucial one, because this option specifies the annotation generation methodology. We simply implement a different Python function for each annotation creation method for the sake of back compatibility. The default value of this option is `0-index-1-diff-full-video`. By using this method, (i) we use the entire videos and (ii) we create foils by incrementing/decrementing the gold count by 1. Each method includes docstrings, so please check the code for more detail. To create annotations with the normalized or the original (unnormalized) dataset, just pass `--normalized` or `--unnormalized` flags (default behaviour is normalized, because I am 100% sure that I would forget to pass this option).


### Annotation Format

 We tried to imitate the VALSE JSON annotations. We performed several changes,

 1. We removed image-related fields.
 2. We added video-related fields: `youtube_id`, `video_file`, `start_time`, `end_time`, `time_unit`.
 3. We also changed `foil` to `foils` to have the ability to work with multiple foils, which makes things harder for the models.


 Here we share a minimal JSON annotation file for this task,

```json
{
    "0-index-1-diff-full-video-000-0-233": {
        "dataset": "QUVA",
        "original_split": "test",
        "dataset_idx": "000",
        "youtube_id": null,
        "video_file": "000_rope_beach.mp4",
        "start_time": 0,
        "end_time": 233,
        "time_unit": "pts",
        "caption": "a man is skipping rope exactly 17 times.",
        "foils": [
            "a man is skipping rope exactly 16 times.",
            "a man is skipping rope exactly 18 times."
        ],
        "foiling_methods": [
            "-1",
            "+1"
        ],
        "template": "a man is skipping rope exactly <number> times.",
        "classes": 17,
        "classes_foils": [
            16,
            18
        ],
        "normalized": true
    },
    "0-index-1-diff-full-video-001-0-606": {
        "dataset": "QUVA",
        "original_split": "test",
        "dataset_idx": "001",
        "youtube_id": null,
        "video_file": "001_trampoline.mp4",
        "start_time": 0,
        "end_time": 606,
        "time_unit": "pts",
        "caption": "a kid is jumping on a trampoline exactly 14 times.",
        "foils": [
            "a kid is jumping on a trampoline exactly 13 times.",
            "a kid is jumping on a trampoline exactly 15 times."
        ],
        "foiling_methods": [
            "-1",
            "+1"
        ],
        "template": "a kid is jumping on a trampoline exactly <number> times.",
        "classes": 14,
        "classes_foils": [
            13,
            15
        ],
        "normalized": true
    }
}
```