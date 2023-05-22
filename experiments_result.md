# Results from experiments

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

2. Using argmax, temp = 1

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

3. int8

```
{
  "exact": 82.70024425166343,
  "f1": 86.23343966074528,
  "total": 11873,
  "HasAns_exact": 80.8029689608637,
  "HasAns_f1": 87.8794920870496,
  "HasAns_total": 5928,
  "NoAns_exact": 84.59209419680404,
  "NoAns_f1": 84.59209419680404,
  "NoAns_total": 5945
}
```

4. int4 (GPTQ)

```
{
  "exact": 81.31895898256549,
  "f1": 85.0672091132973,
  "total": 11873,
  "HasAns_exact": 79.47031039136303,
  "HasAns_f1": 86.97755968322865,
  "HasAns_total": 5928,
  "NoAns_exact": 83.16232127838519,
  "NoAns_f1": 83.16232127838519,
  "NoAns_total": 5945
}
```

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
