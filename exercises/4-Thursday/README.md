# Thursday Exercises: Sequential Models for NLP

## Overview

These exercises apply RNN/LSTM architectures to real NLP tasks. Thursday includes a **collaborative pair programming exercise** that synthesizes the week's concepts.

## Exercise Schedule

| Exercise | Type | Duration | Prerequisites |
|----------|------|----------|---------------|
| 01: RNN Text Generation | Implementation | 60 min | Demo 01, Reading 01 |
| 02: RNN vs LSTM Sentiment (PAIR PROGRAMMING) | Collaborative | 90 min | Demos 02-04, Readings 01-04 |
| 03: Sequence Processing Pipeline | Implementation | 45 min | Demo 04, Reading 04 |

## Prerequisites

- Completed readings: `01-04` in `readings/4-Thursday/`
- Watched demos: `demo_01` through `demo_04`
- Python environment with TensorFlow 2.x, NLTK

## Learning Objectives

By completing these exercises, you will be able to:

1. Build character-level RNNs for text generation
2. Compare RNN and LSTM performance on sentiment analysis
3. Implement proper sequence padding and masking
4. Collaborate effectively using pair programming techniques

## Getting Started

```bash
cd exercises/4-Thursday
pip install -r requirements.txt
```

## Thursday Protocol: Pair Programming

**Exercise 02 is a collaborative activity.** Work with a partner following these rules:

### Roles
- **Driver**: Types the code, focuses on implementation details
- **Navigator**: Reviews code, thinks strategically, catches errors

### Rotation Schedule
- Switch roles every 25 minutes
- Complete handoff: Driver explains current state before switching

### Deliverables
- Working RNN sentiment classifier
- Working LSTM sentiment classifier
- Performance comparison document

## Guidance Philosophy

- **Exercise 01**: Moderate guidance with scaffolding
- **Exercise 02**: Collaborative - roles define guidance level
- **Exercise 03**: Minimal guidance - independent implementation
