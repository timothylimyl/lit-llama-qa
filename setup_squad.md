# Steps to finetune GPT model on SQuAD 2.0

\*Only works for LoRA and Full finetuning

## Step 1: Setup the repo

Clone the repo

```bash
git clone https://github.com/timothylimyl/lit-llama-qa
cd lit-llama-qa
```

Install dependencies

```bash
pip install -r requirements.txt
```

Refer to [the guide](howto/download_weights.md) to download and convert the weights accordingly.

## Step 2: Prepare the data

Step 2a - The raw SQuAD 2.0 data can be found at `generation_qa/squad2.0` namely `train-v2.0.json` and `dev-v2.0.json`. Run:

```
cd generation_qa/squad2.0
python3 convert_format.py
```

Step 2b - Move `squad_data_dev.jsonl` and `squad_data_train.jsonl` into `data/squad2.0`

```
cd ../..  #go back to main folder
cp generation_qa/squad2.0/squad_data_*.jsonl data/squad2
```

Step 2c - Run `python3 scripts/prepare_squad.py` to get our training and dev data in the form of `train.pt` and test.pt`. It will also generate two plots (I have already uploaded it for preview).

## Step 3: Start finetuning

(A) For full finetuning, run `python3 finetune_full_squad.py`

or

(B) For LoRA finetuning, run `python3 finetune_lora_squad.py`

Note: Rename the location you want the weights to be saved into accordingly. For example, `python3 finetune_lora_squad.py --out_dir out/lora/squad2/rank8_experiment`. The finetuned weights will then appear in the `out/lora/squad2/rank8_experiment`.

**Please remember to change the parameters according to your own hardware constraints and rename the folder.**

## Step 4: Official Evaluation on Dev Set

\*Example given for LoRA but same goes for full finetune.

Step 4a - To evaluate, run `python3 evaluate_lora_squad.py --lora_path out/lora/squad2/folder-name/iter-weights-that-you-want-to-evaluate.pt`

Once this is running, you will see all the predictions being printed in your terminal.

Step 4b - All predictions will all be saved as `squad_eval_predict_{log_name}.json` file so that we can run the official evaluation script on it. **`log_name` here takes the name of the iteration and loss of the finetuned weights that you used.**

Step 4c - Copy `squad_eval_predict_{log_name}.json` into `generation_qa/squad2.0` and then run `squad_evaluation.py` following the commands (as per example):

```
cp out/lora/squad2/folder-name/squad_eval_predict_{log_name}.json generation_qa/squad2.0
cd generation_qa/squad2.0
python3 squad_evaluation.py dev-v2.0.json squad_eval_predict_{log_name}.json
```

🎉🎉🎉 Finally, you should expect the results to be printed, example of result format:

```
{
  "exact": 84.85639686684073,
  "f1": 88.12948928646375,
  "total": 11873,
  "HasAns_exact": 80.41497975708502,
  "HasAns_f1": 86.9705509949707,
  "HasAns_total": 5928,
  "NoAns_exact": 89.28511354079058,
  "NoAns_f1": 89.28511354079058,
  "NoAns_total": 5945
}
```
