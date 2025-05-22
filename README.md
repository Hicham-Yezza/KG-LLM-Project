# KG-Enhanced Summarisation Project: Integrating Knowledge Graphs into BART for Improved Factual Consistency

**Author:** Hicham Yezza  
**Year:** 2024  
**Notebook:** [`KG_LLM_Project.ipynb`](https://github.com/Hicham-Yezza/Neurosymbolic-LLM-Project/blob/main/KG_LLM_Project.ipynb)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hicham-Yezza/Neurosymbolic-LLM-Project/blob/main/KG_LLM_Project.ipynb)

---

## Overview

This project implements and evaluates four approaches to enhancing Large Language Model (LLM) summarisation—specifically using BART—by incorporating structured knowledge in the form of Knowledge Graphs (KGs). The notebook covers the entire pipeline, from knowledge graph extraction to summarisation, evaluation, experiment tracking, and fine-tuning.

---

## Motivation

LLMs like BART generate fluent summaries but are prone to factual inconsistency. This project explores whether injecting KG structured knowledge can improve factual grounding in summarisation outputs.

---

## Features

- **Entity & Relation Extraction:** SpaCy-based NER and dependency parsing.
- **Knowledge Graph Construction:** Custom `networkx`-based graphs with filtering heuristics.
- **Four Summarisation Approaches:**
  - `BaseBARTSummarizer`: Standard summarisation with `facebook/bart-large`.
  - `KGEnhancedInputSummarizer`: Structured prompting using KG entities and relations.
  - `GraphAwareAttentionSummarizer`: Custom BART with KG-modulated attention.
  - `KGConsistencyCheckingSummarizer`: Iteratively improves summaries based on KG alignment.
- **Custom BART Variants:** Includes support for graph embeddings and graph-aware attention.
- **Evaluation Framework:** ROUGE, BLEU, METEOR, and BERTScore.
- **Experiment Tracking:** Integrated with Weights & Biases (wandb).
- **Statistical Analysis:** One-way ANOVA and Tukey HSD for comparing models.
- **Fine-tuning Loop:** Full pipeline for training `facebook/bart-large` on a KG-enhanced CNN/DailyMail subset.
- **Unit Tests:** Implemented inline using `unittest` for KG extraction and consistency checking.

---

## Notebook Walkthrough

1. **Setup & Installation**
   - Installs required libraries, including compatible `pyarrow`.
   - Manual runtime restart may be required after reinstalling `pyarrow`.

2. **Knowledge Graph Construction**
   - `KnowledgeGraphExtractor` and `KnowledgeGraphGenerator` classes.
   - SpaCy-based extraction and graph building with `networkx`.

3. **Summariser Implementations**
   - Four summarisation models with modular architecture.
   - Each integrates the KG differently (input prompt, attention, validation, etc.).

4. **Evaluation & Logging**
   - `SummarizerEvaluator` supports ROUGE, BLEU, METEOR, and BERTScore.
   - Experiment logging and visualisation via `wandb`.

5. **Model Comparison**
   - Compares summariser outputs on a subset of the CNN/DailyMail dataset.

6. **Statistical Significance Testing**
   - ANOVA and Tukey HSD implemented to test performance differences.

7. **Fine-tuning Pipeline**
   - Loads and processes KG-enhanced CNN/DailyMail data.
   - Full training loop with AMP, gradient accumulation, and evaluation.

8. **Unit Tests**
   - Inline tests for entity/relation extraction and KG consistency scoring.

---

## Setup Instructions

1. Clone or open the notebook in Colab:
   [Open in Colab](https://colab.research.google.com/github/Hicham-Yezza/Neurosymbolic-LLM-Project/blob/main/KG_LLM_Project.ipynb)

2. Install required libraries (first cell in notebook):
   ```bash
   pip uninstall pyarrow -y
   pip install pyarrow==14.0.1
   pip install wandb rouge_score sacrebleu bert-score spacy networkx datasets pandas tqdm transformers nltk torch evaluate node2vec sentence_transformers
