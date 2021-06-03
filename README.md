# Unsupervised Sentence Embedding Benchmark
This repository hosts the data and the evaluation script for reproducing the results reported in the paper: "[TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning](https://arxiv.org/abs/2104.06979)". This benchmark contains four heterogeous, task- and domain-specific datasets: AskUbuntu, CQADupStack, TwitterPara and SciDocs. For details, pleasae refer to the paper.

## Install
```python
git clone https://github.com/kwang2049/unsupse-benchmark.git
cd unsupse-benchmark
pip install -e .
```

## Usage & Example
```python
from unsupse_benchmark import run
from sentence_transformers import SentenceTransformer  # SentenceTransformer is an awesome library for providing SOTA sentence embedding methods. TSDAE is also integrated into it.
import torch

sbert = SentenceTransformer('bert-base-nli-mean-tokens')

@torch.no_grad()
def semb_fn(sentences) -> torch.Tensor:
    return torch.Tensor(sbert.encode(sentences, show_progress_bar=False))

results, results_main_metric = run(
    semb_fn_askubuntu=semb_fn, 
    semb_fn_cqadupstack=semb_fn,  
    semb_fn_twitterpara=semb_fn, 
    semb_fn_scidocs=semb_fn,
    eval_type='test',
    data_eval_path='../data-eval'
)

assert round(results_main_metric['avg'], 1) == 47.6
```