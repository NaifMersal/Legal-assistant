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
- **regular_m3**: Dense retrieval only with BGE-M3
- **hybrid_m3**: Hybrid retrieval combining dense and sparse strategies
- **m3_reranked**: BGE-M3 with reranking post-processing

## 📁 Files Description

### 📈 Evaluation Reports
*Contains overall metrics and category-wise performance summary*

| File | Model | Configuration | Description |
|------|-------|---------------|-------------|
| `evaluation_report_m3.json` | BGE-M3 | IP indexing | Base model performance |
| `evaluation_report_m3_L2.json` | BGE-M3 | L2 indexing | Base model with L2 distance |
| `evaluation_report_law.json` | BGE-M3-Law | IP indexing | Fine-tuned legal model |
| `evaluation_report_law_L2.json` | BGE-M3-Law | L2 indexing | Legal model with L2 distance |
| `evaluation_report_regular_m3.json` | BGE-M3 | Regular retrieval | Dense retrieval only |
| `evaluation_report_hybrid_m3.json` | BGE-M3 | Hybrid retrieval | Dense + Sparse retrieval |
| `evaluation_report_m3_reranked.json` | BGE-M3 | Reranked | Post-processing optimized |

### 🔍 Detailed Results
*Contains query-by-query results with retrieved documents and scores*

| File | Model | Configuration | Use Case |
|------|-------|---------------|----------|
| `detailed_results_m3.json` | BGE-M3 | IP indexing | Deep analysis of base model |
| `detailed_results_m3_L2.json` | BGE-M3 | L2 indexing | L2 distance detailed results |
| `detailed_results_law.json` | BGE-M3-Law | IP indexing | Legal model query analysis |
| `detailed_results_law_L2.json` | BGE-M3-Law | L2 indexing | Legal + L2 detailed results |
| `detailed_results_regular_m3.json` | BGE-M3 | Regular retrieval | Dense retrieval analysis |
| `detailed_results_hybrid_m3.json` | BGE-M3 | Hybrid retrieval | Dense + Sparse detailed view |
| `detailed_results_m3_reranked.json` | BGE-M3 | Reranked | Reranking impact analysis |

### 📊 Visualization Charts
*Performance metrics visualizations (embedded above)*

| Chart | Model | Configuration | Visual Focus |
|-------|-------|---------------|--------------|
| `evaluation_metrics_m3.png` | BGE-M3 | IP indexing | Base performance visualization |
| `evaluation_metrics_m3_L2.png` | BGE-M3 | L2 indexing | L2 distance metrics |
| `evaluation_metrics_law.png` | BGE-M3-Law | IP indexing | Legal specialization impact |
| `evaluation_metrics_law_L2.png` | BGE-M3-Law | L2 indexing | Legal + L2 performance |
| `evaluation_metrics_regular_m3.png` | BGE-M3 | Regular retrieval | Dense retrieval metrics |
| `evaluation_metrics_hybrid_m3.png` | BGE-M3 | Hybrid retrieval | Dense + Sparse comparison |
| `evaluation_metrics_m3_reranked.png` | BGE-M3 | Reranked | Reranking effectiveness |

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

---

## 📊 Performance Visualizations

### Base Model Comparisons

#### BGE-M3 with Inner Product (IP) Indexing
![BGE-M3 Performance](asstes/evaluation_metrics_m3.png)

#### BGE-M3 with L2 Distance Indexing
![BGE-M3 L2 Performance](asstes/evaluation_metrics_m3_L2.png)

### Fine-tuned Model Comparisons

#### BGE-M3-Law with Inner Product (IP) Indexing
![BGE-M3-Law Performance](asstes/evaluation_metrics_law.png)

#### BGE-M3-Law with L2 Distance Indexing
![BGE-M3-Law L2 Performance](asstes/evaluation_metrics_law_L2.png)

### Advanced Retrieval Strategies

#### Regular Retrieval (Dense Only)
![Regular M3 Performance](asstes/evaluation_metrics_regular_m3.png)

#### Hybrid Retrieval (Dense + Sparse)
![Hybrid M3 Performance](asstes/evaluation_metrics_hybrid_m3.png)

#### Reranked Results
![M3 Reranked Performance](asstes/evaluation_metrics_m3_reranked.png)

---

## 🔍 Model Performance Comparison

| Model Variant | Indexing Method | Retrieval Strategy | Key Strengths |
|---------------|----------------|-------------------|---------------|
| **BGE-M3** | Inner Product | Standard | Baseline performance, good general retrieval |
| **BGE-M3** | L2 Distance | Standard | Alternative similarity metric |
| **BGE-M3-Law** | Inner Product | Standard | Domain-specialized, legal context aware |
| **BGE-M3-Law** | L2 Distance | Standard | Legal specialization + L2 similarity |
| **BGE-M3** | Inner Product | Regular | Dense retrieval only |
| **BGE-M3** | Inner Product | Hybrid | Dense + Sparse combination |
| **BGE-M3** | Inner Product | Reranked | Post-processing optimization |

## Notebooks

- `evaluation_pipeline.ipynb` - Main evaluation pipeline for standard retrieval
- `evaluation_pipeline_with_reranker.ipynb` - Evaluation pipeline with reranking integration

## 🎯 Key Findings & Analysis

### 🔬 Comparative Analysis

The evaluation results enable comprehensive comparison across multiple dimensions:

#### 1. **Model Specialization Impact**
- **Base Model (BGE-M3)**: General-purpose embedding performance
- **Fine-tuned Model (BGE-M3-Law)**: Domain-specific legal document understanding
- **Impact Assessment**: Quantifies the benefit of legal domain fine-tuning

#### 2. **Similarity Metric Comparison**
- **Inner Product (IP)**: Cosine similarity-based retrieval
- **L2 Distance**: Euclidean distance-based retrieval  
- **Performance Differential**: Shows which metric works better for legal documents

#### 3. **Retrieval Strategy Effectiveness**
- **Standard Retrieval**: Baseline dense embedding approach
- **Regular Pipeline**: Dense retrieval only (BGE-M3 embeddings)
- **Hybrid Approach**: Combined dense and sparse retrieval strategies
- **Reranking**: Post-processing optimization with re-scoring

### 📊 Quick Reference

For detailed analysis:
- **📈 Summary Metrics**: Check `evaluation_report_*.json` files
- **🔍 Query Analysis**: Examine `detailed_results_*.json` files  
- **📊 Visual Comparison**: Review the performance charts above

### 🚀 Usage Recommendations

Based on the evaluation results, you can:
1. **Select the best-performing model** for your legal document retrieval needs
2. **Choose optimal indexing strategy** (IP vs L2) for your use case
3. **Implement the most effective retrieval approach** (standard, hybrid, or reranked)

---

*For implementation details, refer to the Jupyter notebooks in the parent directory.*
