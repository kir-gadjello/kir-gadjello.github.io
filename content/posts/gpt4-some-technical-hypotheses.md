---
title: "GPT-4 architecture: what we can deduce from research literature"
date: 2023-03-14T00:00:00+00:00
weight: 1
aliases: ["/gpt4-some-technical-hypotheses"]
tags: ["futurology", "technical", "ml", "lm", "sparsity", "attention", "efficiency", "gpt", "ai"]
author: "Kirill Gadjello"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Which techniques are likely to land in the latest foundation model? [V0.1]"
canonicalURL: "https://kir-gadjello.github.io/gpt4-some-technical-hypotheses"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "/chinchilla_and_llama_scaling.png" # image path/url
    alt: "a desirable scaling curve - deepmind's chinchilla to the left, meta's llama to the right" # alt text
    caption: "Machine Learning as a field is very explicit about the research KPIs. Left: Deepmind's Chinchilla, Right: Meta's LLaMA"
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
editPost:
    URL: "https://github.com/kir-gadjello/kir-gadjello/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

*This text is my personal opinion, developed by researching publicly available sources such as [research publications](https://arxiv.org/list/cs.LG/recent) and rumors. I did not and do not work in any of the companies whose current or future products this text speculates about.*

Intended audience: people with engineering experience or some basic ML knowledge who are interested in language modeling techniques that may have been selected for implementation by "GPT-4" authors from OpenAI. We need such speculation, because the authors have elected to keep the technical detail private, citing safety concerns and competitive landscape. The bulk of this text had been written in the first days of March, when actual capabilities of GPT-4 remained an enigma.

**TL;DR of hypotheses developed in this post** (note that some references postdate the GPT-4 pretraining run; but they either have precedents in literature, or are assumed to be known privately in advance of publication, [as has been the case with chinchilla scaling law, for example](https://time.com/6246119/demis-hassabis-deepmind-interview/))):

* *(High confidence)* **Intra-tensor sparsity** from [Google's Scaling Transformer and Terraformer](https://arxiv.org/abs/2111.12763). This alone provides an OOM inference performance improvement, allowing the system to break through the memory bandwidth wall.
* *(High confidence)* **MoE with routing**, similar or more advanced than in [Google's switch Transformer](https://arxiv.org/abs/2101.03961) [2](https://arxiv.org/abs/2202.08906) (i.e. coarse-grained FFN layer sparsity). This improves inference performance in synergy with the former technique.
* *(High confidence)* Some variant of **efficient attention**; could be just an LSH attention from [Terraformer](https://arxiv.org/abs/2111.12763) or a more conservative parameter-efficient convolutional attention from Scaling Transformer from the same paper. In the base case it just improves inference performance in synergy with the former techniques, but in the more interesting case it might also improve asymptotic performance - paving the way to 8k...32k context window size at a bearable cost.
* *(High confidence)* [**UL2 pretraining objective**](https://arxiv.org/abs/2205.05131) for improved scaling law (note how we haven't seen the UL2-Chinchilla just yet).
* *(Confirmed)* **"Image-enabled multimodality"** similar to [Aleph's lumionous](https://www.aleph-alpha.com/luminous), but with heavy prioritizing of plaintext over image modality, with the vast majority of pretraining compute spent on processing text tokens - [as neccessiated by lackluster multimodal scaling laws](https://arxiv.org/abs/2301.03728). The pretraining could even be text-only, with [vision bolted on later during finetuning](https://arxiv.org/abs/2301.12597).
* *(High confidence)* **A new dataset with on the order of 10 trillion tokens**, with significant part of it consisting of licensed material (could be youtube transcripts computed with help of Whisper and a large amount of books).
* *(High confidence)* [**muTransfer for efficient hyperparameter estimation.**](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/)
* *(Moderate confidence)* [**Advanced positional encoding**](https://arxiv.org/abs/2212.10554) for less attention noise and runtime scaling beyond the context window size used for pretraining. Notably, also applied in Microsoft's [KOSMOS-1](https://arxiv.org/abs/2302.14045).
* *(Moderate confidence)* **Encoder-Decoder architecture**, for better scaling and UL2 synergy. This could explain the special nature of "system" prompt.
* *(Moderate confidence)* **Retrieval-augmented language modeling** for enhanced scaling (see [Meta's ATLAS](https://arxiv.org/abs/2208.03299)).
* *(Low confidence)* Some [**nontrivial form of dataset pruning and/or reordering**](https://arxiv.org/abs/2206.14486) to maximize pretraining sample efficiency and enhance the resulting scaling law.

This list doesn't include engineering-heavy post-training enhancements; I simply expect [world-class ML engineering with all the best model compression and inference optimization techniques stacked together](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).

Below is my interpretation of the recent history and incentives in this burgeoning field, and how I came to these guesses.

## Introduction

* As of 2023, Language Models (LMs) have claimed an ever-growing amount of attention across wide swathes of society: groups as different as enthusiastic hackers, public intellectuals, corporate strategy execs and VC investors all have some stake in the future of LMs.
* The current trajectory of LM progress depends on four pillars:
    1. **Scaling**: few entities investing significant resources (examples: [1](https://news.microsoft.com/source/features/ai/openai-azure-supercomputer/) [2](https://techcrunch.com/2022/05/11/google-launches-a-9-exaflop-cluster-of-cloud-tpu-v4-pods-into-public-preview/) [3](https://about.fb.com/news/2022/01/introducing-metas-next-gen-ai-supercomputer/)) to make training runs of novel large models possible. Notably, these few entities serve as gatekeepers, deciding which model architectures, on which datasets and for which applications are trained at all. The basis for these decisions is multi-factored, with risk considerations (technical, reputational) being a significant factor. We will pay particular attention to technical risks of including some techniques in the training run and, on the other hand, to de-risking - that is, **large-scale validation of techniques** either originating in research publications or privately known.
    2. **Algorithmic progress**: research on improving compute-, parameter efficiency and expanding the applicability of deep learning to novel modalities, applications and abilities. Notably, this is a minor subset of DL research, seeing as much of it doesn't prove worthy of scaling (see below) - which in turn is a subset of ML research. Some approaches to measure algorithmic progress study combinations of enhancements for some fixed problem or model architecture: [AI and Efficiency (OpenAI)](https://openai.com/research/ai-and-efficiency), [Deep Neural Nets: 33 years ago and 33 years from now (Karpathy's blog)](http://karpathy.github.io/2022/03/14/lecun1989/), [EfficientQA challenge (Google)](https://ai.google.com/research/NaturalQuestions/efficientqa), [MosaicML Composer documentation](https://docs.mosaicml.com/en/stable/).
    3. **Engineering**: broadly understood as the development of systems making the first two pillars possible at scale, such as deep learning frameworks [DLF1](https://en.wikipedia.org/wiki/PyTorch) [DLF2](https://github.com/google/jax), [parallelism schemes](https://arxiv.org/abs/1806.03377) [DLPS2](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and associated [engines](https://arxiv.org/abs/2301.08984) and [compilers](https://www.usenix.org/conference/osdi20/presentation/ma).
    4. **Hardware**: arguably could be merged with engineering, but it is sufficiently insulated to be its own field and professional community, and in a certain long-term sense it guides the evolution of the former three pillars - some academics have coined the uncharitable term ["hardware lottery"](https://arxiv.org/abs/2009.06489) to describe this path-dependent dynamic.
    5. And what looks poised to become a fifth pillar: [impending regulation](https://twitter.com/sama/status/1635136281952026625).

Just a few years ago this structure was not yet in place, likely owing to **"inertia"** associated with fundamental technical and economical risks of investing significant (by the standards of usual software engineering or academic ML) resources in this unproven field.

This **inertia** had been broken by a few companies and associated private research organizations. Arguably, the most important events that changed the outlook for this nascent industry were [unprecedented research milestones, demonstrating reliable improvement of model performance with scaling data, parameters and compute](https://arxiv.org/abs/2005.14165). For the first time it was shown that training ever larger language models could lead to ever more impressive tools with emergent functionality responsive to natural language input provided at runtime. It was also demonstrated that large enough language models trained on general datasets can be quickly *finetuned* to become competent at specific tasks. After those original publications and a tech demo, an exponential stream of novel milestones (DALL-E, Jukebox, GLIDE, Chinchilla, Flamingo, etc) continues to de-risk entire new subfields, making them attractive avenues for investment by more cautious competitors. Something could be said here about comparative novelty and priority of Google/Deepmind's and OpenAI's research and perhaps the unfair amount of attention and industry leadership OpenAI gets with its publications.

So far, most of scaling DL research, especially in language modeling, has been done with *dense Transformer* variants - and while we have seen the algorithmic progress in objective function (UL2, FCM), optimizers (Lion) and some small tweaks of the base architecture, it mostly remained in place, at least in experiments that were tried at scale. Some smaller-scale work is more interesting (H3, Hyena, Kosmos). Obviously, not every promising and technically interesting technique proves its potential at scale, or even should be scaled.

The takeaway is that, very likely, the "GPT-4" model is -- in sharp contrast to its predecessors -- implemented on top of deep modifications of the base [Transformer (Vaswani 2017)](https://arxiv.org/abs/1706.03762) architecture, setting up new standard of parameter- and inference compute efficiency and making [Deepmind's Chinchilla](https://arxiv.org/abs/2203.15556) and [Meta's LLaMA](https://arxiv.org/abs/2302.13971) obsolete, as they have made obsolete the preceding scaling laws [by Kaplan et al](https://arxiv.org/abs/2001.08361). I offer an educated guess on specific technical methods by which this was achieved.

{{< rawhtml >}}<center><h1>---{ Under construction }---</h1></center>{{< /rawhtml >}}

## What we know
### Incentive structure for training a large generative model

If you are a **private research lab** (Deepmind, Google Brain, Meta Research) owned by an incumbent corporation, you optimize for some mix of the following:
* Comparability with baselines (notably, LLaMA authors wrote that they applied basic transformer architecture in lieu of more efficient alternatives to keep their project comparable to Deepmind's Chinchilla).
* Accuracy on academic benchmarks.
* Scientific novelty.
* Accuracy on parent company's problems (with or without finetuning).

If you are a **for-profit product-oriented organization** like OpenAI, your incentives are different in important ways:
* You don't care much about comparability with baselines -- you are free to stack powerful enhancements to see where this takes you.
* You care less about academic benchmarks (although it's good for PR to beat them).
* You care a lot about delivering a successful product -- a model useful for 0-shot execution of customer's tasks that can also be finetuned by client startups to provide novel profitable products to end-users.
* You can and do optimize your foundational model architecture for inference efficiency and compatibility with advanced engineering optimizations, as you care a lot about inference economy.

### Deep learning trick stacking: most tricks don't scale
A large part of published DL papers deals with some methods of enhancing the model's performance.
The most important takeways here are:
1. *Only a small ratio of published "tricks" end up contributing to enhanced language model performance at scale.*
2. *This necessitates costly experiments to de-risk promising enhancements (hopefully done by your competitor, i.e. Google, and published in the open), or schemes to study these techniques in smaller models, or even falling back to subjective selection criteria.*

### Fundamentals: the chinchilla[1,2,3] scaling laws
### Multimodality is a desirable feature, but doesn't help language model pretraining -- at least with current architectures and medium scale
### Immense pressure to optimize inference efficiency
### "DV" product pricing and technical limitations
### The chinchilla rationale behind dataset growth
### Training and inference hardware available
## Tiered nomenclature of model enhancements
### "Foundational" - requiring training a foundation model from scratch ($$$)
### "Addons" - which can be applied later during finetuning ($)
### "Engineering" - significant development time by world-class specialists ($$)
## A list of novel techniques I expect GPT-4 to show at scale
### Intra-tensor Sparsity (Scaling Transformer & Terraformer)
### Layer Sparsity (MoE)
### UL2 objective function
### Efficient attention
### Retrieval
### Encoder-Decoder design
## Unlikely enhancements and other "black swans"
### Novel convolutional attention
### Dataset pruning
## Closing thoughts and musings on technological trajectory
##  Useful References

0. (Lilian Weng's blog) [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
1. (Lilian Weng's blog) [The Transformer Family Version 2.0
](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
1. [Sparse is Enough for scaling transformers](https://arxiv.org/abs/2111.12763)
2. [Unifying language modeling paradigms](https://arxiv.org/abs/2205.05131)
3. [ST-MoE: Designing Stable and Transferable Sparse Expert Models
](https://arxiv.org/abs/2202.08906)
