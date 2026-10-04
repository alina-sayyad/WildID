# WildID

## Individual Wildlife Re-Identification and Conservation Intelligence System

WildID is an evidence-oriented wildlife re-identification prototype designed to answer a key conservation question:

> **Which individual animal is this?**

Wildlife monitoring systems can detect that an animal is present and identify its species, but determining whether the same individual has been observed before is a different challenge.

WildID focuses on this individual-level re-identification problem by comparing a query wildlife image against a gallery of known individuals, ranking the closest candidates, and allowing the system to return **No Reliable Match** when the available evidence is insufficient.

## Current Prototype

The current working prototype focuses on **tiger individual re-identification**.

The R001 baseline demonstrates:

- Wildlife-specific visual embeddings using **MegaDescriptor-T-224**
- Query-to-gallery similarity retrieval
- Candidate ranking using cosine similarity
- Top-5 candidate retrieval
- Evidence-based match decision
- Explicit **No Reliable Match** state
- Judge-facing Streamlit interface

The repository contains only the artifacts required for the demonstration. The full training dataset is not included.

## R001 Baseline Results

The R001 experiment was evaluated on an ATRW-based validation split.

| Metric | Result |
|---|---:|
| Top-1 Accuracy | **97.75%** |
| Top-5 Accuracy | **99.44%** |
| mAP | **94.22%** |
| Identities | **107** |
| Gallery Images | **642** |
| Query Images | **1,245** |

These are measured offline results from the evaluated R001 split.

They demonstrate that the current prototype can perform individual-level retrieval on the evaluated data. They should **not** be interpreted as production accuracy or field deployment validation.

## Decision Logic

The prototype uses cosine similarity between the query embedding and gallery embeddings.

For the current demonstration, a similarity threshold of **0.90** is used for the final decision:

- **MATCH** — strongest candidate reaches the prototype threshold
- **NO RELIABLE MATCH** — strongest candidate does not reach the threshold

The threshold is an experimental prototype choice and would require additional species-specific and field validation before operational deployment.

## System Concept

```text
Wildlife Image
      ↓
Visual Embedding
      ↓
Gallery Retrieval
      ↓
Candidate Ranking
      ↓
Similarity Evidence
      ↓
Match / No Reliable Match
