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

### Meeting with Albert

We also met with Albert on 25th October. We fixed syntax of the textual templates all together, and went over the semantics / the meanings of the constructed templates. While fixing the templates, we obeyed the rules of British English (e.g. skipping rope instead of jumping rope). We also switched to present tense by abandoning present continious tense, as suggested by Albert. We also replaced specific terminology with more general expressions (e.g. use lifting weights instead of doing skullcrushers). Lastly, we also discussed how to create


## Generating Annotations

This part describes how we semi-automatically create the final JSON annotations consumed by the models using the templates.