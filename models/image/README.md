# Running CLIP and BLIP2
CLIP Paper: https://arxiv.org/abs/2103.00020
BLIP2 Paper: https://arxiv.org/abs/2301.12597

First, setup the environment by following [these steps](../../README.md). After you prepare the environment, run the following command,

```bash
python run_{clip,blip2}.py \
    --input-file /path/to/annotations.json \
    --quva-dir /path/to/quva \
    --something-something-dir /path/to/dataset \
    --youtube-dir /path/to/youtube-videos \
    --output-file /path/to/output.json
```

This command will produce a results annotation file `/path/to/output.json`. To generate scores for the proficiency task, pass the `--proficiency` flag.