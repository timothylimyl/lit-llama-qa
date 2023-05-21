<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/Lit_LLaMA_Badge3x.png" alt="Lit-LLaMA" width="128"/>
</div>

# ⚡ Purpose of Lit-LLaMA-QA ⚡

Experimenting with fine-tuning Generative Pre-trained Model (GPT) by using NLP academic dataset for evaluation to get better intuition and knowledge on how to best fine-tune GPT models and understand GPT model's performance before training it for abstractive tasks that requires human evaluation.

**Goal 1: By using academic dataset, we can get some intuition on how to improve fine-tuning and understand what works.** For example, to answer questions such as "Does LoRA really work?" is very difficult with generative responses as human evaluation is challenging and time-consuming. We want to get grounded feedback on the proposed training methodology. Thus, we will rely on using academic dataset first to get some intuition on practices to follow.

**Goal 2: To gauge how performant are GPT models, especially under PeFT methods**. With academic dataset, we at least have some baseline results while experimenting with different methods. We are also curious on how easy would it be to reach SOTA results.

We are focusing on QA dataset first as the future goal is to train abstractive qa with dialogue based replies (hard to evaluate, no standard benchmark for this). To start off, our targeted dataset will be SQuAD 2.0 which is an extractive dataset with unanswerable questions.

Please jump to [Current takeaways from experiments](#current-takeaways-from-experiments) for some of our learnings from experimenting with GPT models.

## SQuAD 2.0

(A) Dataset detail

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable [SQuAD 2.0 reference](https://arxiv.org/pdf/1806.03822.pdf).

Dataset consist of 150,000 and 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. Dev dataset and official evaluation script is provided for evaluation.

(B) Metric

Exact match and F1 score

## Experiments

Experiments is done without tweaking parameters. Results are provided without "bell or whistle", we have not done anything extra to boost the results such as ensembling (generation/model), probability thresholding on unanswerable, etc.

Evaluation is done via [official evaluation script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).

Model: [LLaMA](https://arxiv.org/pdf/2302.13971.pdf) 7B with context length of 512 (float16) unless stated otherwise.

For instructions to set up fine-tuning and replicating our experiment for SQuAD 2.0 dataset, view [`setup_squad.md`](setup_squad.md).

### Experiment 1: LoRA (Rank 8)

1. Using multinomial, temp = 1

```
{
  "exact": 79.58392992504001,
  "f1": 84.2160752961748,
  "total": 11873,
  "HasAns_exact": 75.37112010796221,
  "HasAns_f1": 84.64869466792929,
  "HasAns_total": 5928,
  "NoAns_exact": 83.78469301934399,
  "NoAns_f1": 83.78469301934399,
  "NoAns_total": 5945
}
```

2. Using argmax

```
{
  "exact": 83.2729722900699,
  "f1": 86.67488213657204,
  "total": 11873,
  "HasAns_exact": 81.08974358974359,
  "HasAns_f1": 87.90331909708533,
  "HasAns_total": 5928,
  "NoAns_exact": 85.44995794785534,
  "NoAns_f1": 85.44995794785534,
  "NoAns_total": 5945
}
```

We have tested for a few weights and argmax performance is always superior which is expected as this is an extractive task (answer is found in context). **All generation for extractive QA will be done via argmax.**

### Experiment 2: Full-finetuning

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

Note: Got room for improvement in training, our validation interval to save checkpoint was set too big due to storage concern as each weight saved is 14GB (7B model). It ended up with only the very first checkpoint being saved as all validation loss after the first validation interval was higher.

### Experiment 3: LoRa (Rank 8) - 30B

```
{
  "exact": 87.56843257811842,
  "f1": 90.14054761949711,
  "total": 11873,
  "HasAns_exact": 82.86099865047234,
  "HasAns_f1": 88.01260490659415,
  "HasAns_total": 5928,
  "NoAns_exact": 92.26240538267452,
  "NoAns_f1": 92.26240538267452,
  "NoAns_total": 5945
}
```

### Hardware Requirement

Tested with micro batch size of 1 and batch size of 128 using Gradient Accumulation.

1. LLaMA 7B with context length of 512

- LoRA: ~20GB
- Full-finetuning (FSDP): ~80GB (Single GPU, 80GB A100) but does not work on ~88GB (4x GPU, 22GB A5000)

2. LLaMA 13B with context length of 512

- LoRA: ~40GB

3. LLaMA 30B with context length of 512

- LoRA: ~75GB

## Academic Paper Results and comparison (SQuAD 2.0)

For comparison, we should only compare to the best research out there to get some idea on how good is the performance of fine-tuning llama. Comparison made is for the dev set (as per paper and our own experiment)

| Model                                                          | F1    | Reference     |
| -------------------------------------------------------------- | ----- | ------------- |
| Ours (7B)                                                      | 88.13 | Full-finetune |
| Ours (30B)                                                     | 90.14 | LoRA          |
| [FLAN 137B](https://arxiv.org/pdf/2109.01652.pdf)              | 43.1  | 3-shot        |
| [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)                  | 69.8  | 16-shot       |
| [BERT](https://arxiv.org/pdf/1810.04805.pdf)                   | 83.1  | Supervised    |
| [Retrospective Reader](https://arxiv.org/pdf/2001.09694v4.pdf) | 91.3  | Supervised    |
| [DeBERTa](https://openreview.net/pdf?id=XPZIaotutsD) (large)   | 90.7  | Supervised    |
| [DeBERTa](https://openreview.net/pdf?id=XPZIaotutsD) (base)    | 86.2  | Supervised    |
| [DeBERTa V3](https://arxiv.org/pdf/2111.09543.pdf)             | 91.16 | Supervised    |

DeBERTa V3 paper claims is that F1 score is 91.5. However, **current best on dev set [verified by paperswithcode](https://paperswithcode.com/sota/question-answering-on-squad-v2)** is `deepset/deberta-v3-large-squad2` with F1: 91.16. However, the official eval script (the one we are using) gives a slightly lower result on their model, [refer to Hugging Face repo](https://huggingface.co/deepset/deberta-v3-large-squad2).

Model that was specifically developed / more suitable (architecture,ablations studies) for the task of extractive QA (ex: SQuAD 2.0):

1. BERT
2. Retro-Reader
3. DeBERTa
4. DeBERTa V3

# Current takeaways from experiments

1. Fine-tuning small LM models on consumer GPU is possible via LoRA.

2. LoRA and full-finetuning results achieved can be competitive with SOTA models built for specific tasks.

3. Full fine-tuning results is proven to be better than LoRA for small language model. [LoRA paper](https://arxiv.org/pdf/2106.09685.pdf) claims: "LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters". This claim may not translate too well to smaller models as per our experiment. We need to determine whether is the tradeoff of performance versus training cost and time worth it.

4. We also should take note of hardware constraint. Given models above 7B params, full finetuning may not be feasible for most people due to GPU VRAM requirement.

5. GPT results is amazing considering that GPT models (decoder-only) task is to generate the next token which is not suitable for extractive QA when compared to BERT based model (encoder-only) that can directly classify the start and end token of the context.

6. Fine-tuning GPT models is easy to set up and loss converges pretty fast. Most experiments took just a few hours to 2 days to achieve its lowest validation loss. **For example, fine-tuning the 30B Model using LoRa on 2x80GB A100 (DDP) only took us approximately 5 hours to reach the lowest validation loss.**

# Future Work

1. Finetune for abstractive question and answering under the context length of 2048. This model will be more suitable for real world application.

If time permits:

2. Try bigger language models

- Experiments with the 13B, 30B, 65B variant

3. Experiment with more PeFT techniques

- LoRa with different rank
- Prefix-tuning
- Adapters
- Joining up the ideas (LoRA + Adapter + Prefix-tuning)

4. Fine-tuning the LM directly for Unified QA then evaluation can be done with every QA dataset, [paper inspiration](https://arxiv.org/pdf/2202.12359.pdf).
