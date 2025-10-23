# Evaluation Pipeline

This directory contains the evaluation results and pipelines for the Legal Assistant's information retrieval system. The evaluation compares different embedding models and indexing approaches to determine the best configuration for legal document retrieval.

## Overview

The evaluation pipeline assesses the performance of retrieval systems using various metrics including Recall@K, Precision@K, NDCG@K, Hit Rate@K, MRR (Mean Reciprocal Rank), and MAP (Mean Average Precision).

## Embedding Models

### 1. BGE-M3 (Base Model)
- **Model**: `BAAI/bge-m3`
- **Description**: The standard BGE-M3 (BAAI General Embedding - Multi-lingual, Multi-functionality, Multi-granularity) model
- **Files prefix**: `m3`

### 2. BGE-M3-Law (Fine-tuned Model)
- **Model**: BGE-M3 fine-tuned on law datasets
- **Description**: A specialized version of BGE-M3 that has been fine-tuned specifically for legal domain documents
- **Files prefix**: `law`

## Indexing Approaches

Two different similarity/distance metrics are used for FAISS indexing:

### 1. IP (Inner Product)
- **Description**: Inner Product similarity metric (default)
- **Files**: Files without `_L2` suffix use IP indexing
- **Use case**: Better for normalized embeddings and cosine similarity

### 2. L2 (Euclidean Distance)
- **Description**: L2 distance (Euclidean distance) metric
- **Files**: Files with `_L2` suffix
- **Use case**: Better for absolute distance measurements

## Retrieval Variants

### Standard Retrieval
- **m3**: BGE-M3 with IP indexing
- **m3_L2**: BGE-M3 with L2 indexing
- **law**: BGE-M3-Law with IP indexing
- **law_L2**: BGE-M3-Law with L2 indexing

### Advanced Retrieval
- **regular_m3**: Regular retrieval approach with BGE-M3
- **hybrid_m3**: Hybrid retrieval combining multiple strategies
- **m3_reranked**: BGE-M3 with reranking post-processing

## Files Description

### Evaluation Reports
Contains overall metrics and category-wise performance:
- `evaluation_report_m3.json` - BGE-M3 with IP indexing
- `evaluation_report_m3_L2.json` - BGE-M3 with L2 indexing
- `evaluation_report_law.json` - BGE-M3-Law with IP indexing
- `evaluation_report_law_L2.json` - BGE-M3-Law with L2 indexing
- `evaluation_report_regular_m3.json` - Regular retrieval variant
- `evaluation_report_hybrid_m3.json` - Hybrid retrieval variant
- `evaluation_report_m3_reranked.json` - Reranked results variant

### Detailed Results
Contains detailed query-by-query results with retrieved documents:
- `detailed_results_m3.json` - BGE-M3 with IP indexing
- `detailed_results_m3_L2.json` - BGE-M3 with L2 indexing
- `detailed_results_law.json` - BGE-M3-Law with IP indexing
- `detailed_results_law_L2.json` - BGE-M3-Law with L2 indexing
- `detailed_results_regular_m3.json` - Regular retrieval variant
- `detailed_results_hybrid_m3.json` - Hybrid retrieval variant
- `detailed_results_m3_reranked.json` - Reranked results variant

### Visualization Charts
Performance metrics visualizations:
- `evaluation_metrics_m3.png` - BGE-M3 with IP indexing
- `evaluation_metrics_m3_L2.png` - BGE-M3 with L2 indexing
- `evaluation_metrics_law.png` - BGE-M3-Law with IP indexing
- `evaluation_metrics_law_L2.png` - BGE-M3-Law with L2 indexing
- `evaluation_metrics_regular_m3.png` - Regular retrieval variant
- `evaluation_metrics_hybrid_m3.png` - Hybrid retrieval variant
- `evaluation_metrics_m3_reranked.png` - Reranked results variant

## Evaluation Metrics

### Recall@K
Measures the proportion of relevant documents retrieved in the top K results.

### Precision@K
Measures the proportion of retrieved documents that are relevant in the top K results.

### NDCG@K (Normalized Discounted Cumulative Gain)
Measures ranking quality, considering both relevance and position of retrieved documents.

### Hit Rate@K
Measures whether at least one relevant document appears in the top K results.

### MRR (Mean Reciprocal Rank)
Average of the reciprocal ranks of the first relevant document for each query.

### MAP (Mean Average Precision)
Mean of average precision scores across all queries.

## Notebooks

- `evaluation_pipeline.ipynb` - Main evaluation pipeline for standard retrieval
- `evaluation_pipeline_with_reranker.ipynb` - Evaluation pipeline with reranking integration

## Key Findings

The results allow comparison between:
1. **Base vs Fine-tuned**: BGE-M3 vs BGE-M3-Law to assess the impact of domain-specific fine-tuning
2. **IP vs L2 Indexing**: Different similarity metrics for retrieval
3. **Retrieval Strategies**: Standard, hybrid, and reranked approaches

Refer to the individual JSON reports for detailed metrics and the PNG files for visual comparisons.
