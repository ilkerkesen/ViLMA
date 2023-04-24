# Running GPT-2 and OPT
First, setup the environment by following [these steps](../../README.md). After you prepare the environment, run the following command,

```bash
python models/text/run.py \
    -i /path/to/annotations.json \
    -o /path/to/output.json \
    --model-name {gpt2,facebook/opt-2.7b,facebook/opt-6.7b}
```

This command will produce a results annotation file `path/to/output.json`. To generate scores for the proficiency task, pass the `--proficiency` flag. Execute `python models/text/run.py --help` to see other options.