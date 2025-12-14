# Wednesday Exercises: NLP Foundations

## Overview

These exercises reinforce tokenization, text encoding, and word embeddings covered in today's demos. You'll build text preprocessing pipelines and explore semantic relationships captured by embeddings.

## Exercise Schedule

| Exercise | Type | Duration | Prerequisites |
|----------|------|----------|---------------|
| 01: Tokenization Pipeline | Implementation | 45 min | Demo 01, Readings 01-02 |
| 02: One-Hot vs. Dense Encoding | Implementation | 45 min | Demo 02, Reading 03-04 |
| 03: Embedding Explorer | Implementation | 60 min | Demo 03, Reading 05 |

## Prerequisites

- Completed readings: `01-05` in `readings/3-Wednesday/`
- Watched demos: `demo_01` through `demo_03`
- Python environment with TensorFlow, NLTK, Gensim (optional)

## Learning Objectives

By completing these exercises, you will be able to:

1. Build robust tokenization pipelines for text data
2. Compare one-hot encoding limitations with dense embeddings
3. Explore semantic relationships using Word2Vec-style embeddings
4. Integrate Keras Embedding layers into NLP models

## Getting Started

```bash
cd exercises/3-Wednesday
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

Start with Exercise 01 and proceed in order.

## Guidance Philosophy

- **Exercise 01**: Heavily guided with step-by-step implementation
- **Exercise 02**: Moderate guidance with comparison framework
- **Exercise 03**: Minimal guidance - independent exploration
