from typing import Any, Dict, List, Optional, Tuple, Callable
import json
import os
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# --- Data Structure for Metrics ---

@dataclass
class EvaluationMetrics:
    """A data container for holding all computed evaluation metrics."""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    map_score: float
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]


# --- Combined Evaluator and Experiment Class ---

import os
import json
import random
import itertools
from pprint import pprint
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm

@dataclass
class EvaluationMetrics:
    """A data container for holding all computed evaluation metrics."""
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    map_score: float
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]

class RetrievalEvaluator:
    """
    Loads a QA dataset, runs retrieval evaluation, and stores results.
    
    This class combines the evaluation logic with metadata about the
    specific approach being tested (e.g., model name, approach description).
    """

    def __init__(
        self,
        # Metadata from EvaluationApproach
        name: str,
        approach_desc: str,
        model_name: str,
        
        # Config from RetrievalEvaluator
        retrieve_function: Callable,
        qa_pairs: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10, 20],
        is_baseline: bool = False
    ):
        """
        Initializes the Evaluator.

        Args:
            name: A short, unique name for this evaluation (e.g., "hybrid_run_1").
            approach_desc: A human-readable description (e.g., "hybrid (dense + sparse)").
            model_name: The name of the model(s) used (e.g., "BAAI/bge-m3 + BM25").
            retrieve_function: The function to call to get results for a query.
                               Expected signature: retrieve_function(query: str, k: int, **kwargs) -> (scores, indices)
            qa_pairs: List of QA dictionaries.
            k_values: A list of k values to use for @k metrics.
            is_baseline: Mark this as the baseline for comparisons.
        """
        self.name = name
        self.approach_desc = approach_desc
        self.model_name = model_name
        self.is_baseline = is_baseline
        
        self.retrieve_function = retrieve_function
        self.k_values = k_values
        self.max_k = max(k_values)
        
        self.qa_pairs = qa_pairs
        # Placeholders for results
        self.metrics: Optional[EvaluationMetrics] = None
        self.detailed_results: Optional[List[Dict]] = None

    # --- Core Metric Calculations ---

    def recall_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Recall@k."""
        if not relevant:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k & set(relevant))
        return hits / len(relevant)

    def precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Precision@k."""
        if k == 0:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        hits = len(retrieved_at_k & set(relevant))
        return hits / k

    def reciprocal_rank(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Reciprocal Rank."""
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    def average_precision(self, retrieved: List[int], relevant: List[int]) -> float:
        """Calculate Average Precision."""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        hits = 0
        precision_sum = 0.0
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                hits += 1
                precision_at_i = hits / rank
                precision_sum += precision_at_i
                
        if hits == 0:
            return 0.0
            
        return precision_sum / len(relevant)

    def ndcg_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain (nDCG)@k."""
        relevant_set = set(relevant)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2) # rank = i + 1
        
        num_relevant = len(relevant)
        ideal_k = min(num_relevant, k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
        
        return dcg / idcg if idcg > 0 else 0.0

    def hit_rate_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calculate Hit Rate@k."""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        retrieved_at_k = set(retrieved[:k])
        
        return 1.0 if len(retrieved_at_k & relevant_set) > 0 else 0.0

    # --- Evaluation Execution ---

    def evaluate_single_query(self, qa_pair: Dict[str, Any], retrieve_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            qa_pair: A dictionary containing 'question' and 'relevant_ids'.
            retrieve_kwargs: A dictionary of extra keyword arguments to pass
                             to the self.retrieve_function.
        """
        question = qa_pair.get('question', qa_pair.get('query'))
        relevant_ids = qa_pair.get('relevant_ids', qa_pair.get('references_ids'))
        
        if not question or relevant_ids is None:
            print(f"Skipping malformed qa_pair: {qa_pair.get('id', 'Unknown ID')}")
            return None

        # *** MODIFIED LINE ***
        # Pass retrieve_kwargs to the retrieval function
        scores, retrieved_indices = self.retrieve_function(
            question, 
            k=self.max_k, 
            **retrieve_kwargs
        )
        
        if hasattr(scores, 'tolist'):
             scores_list = scores.tolist()
        else:
             scores_list = list(scores)
             
        if hasattr(retrieved_indices, 'tolist'):
            retrieved_ids = retrieved_indices.tolist()
        else:
            retrieved_ids = list(retrieved_indices)


        results = {
            'qa_id': qa_pair.get('id', 'N/A'),
            'question': question,
            'relevant_ids': relevant_ids,
            'retrieved_ids': retrieved_ids,
            'scores': scores_list,
            'metrics': {}
        }

        for k in self.k_values:
            if k <= self.max_k:
                results['metrics'][f'recall@{k}'] = self.recall_at_k(
                    retrieved_ids, relevant_ids, k)
                results['metrics'][f'precision@{k}'] = self.precision_at_k(
                    retrieved_ids, relevant_ids, k)
                results['metrics'][f'ndcg@{k}'] = self.ndcg_at_k(
                    retrieved_ids, relevant_ids, k)
                results['metrics'][f'hit_rate@{k}'] = self.hit_rate_at_k(
                    retrieved_ids, relevant_ids, k)

        results['metrics']['reciprocal_rank'] = self.reciprocal_rank(
            retrieved_ids, relevant_ids)
        results['metrics']['average_precision'] = self.average_precision(
            retrieved_ids, relevant_ids)

        return results
    
    def evaluate_all(
        self, 
        sample_size: int = None, 
        sample_indices_path: str = None,
        retrieve_kwargs: Dict[str, Any] = None
    ) -> Tuple[EvaluationMetrics, List[Dict]]:
        """
        Evaluate all queries in the dataset and store results in self.metrics.

        Args:
            sample_size: Number of queries to randomly sample for evaluation.
            sample_indices_path: Path to load/save a fixed set of sample indices.
            retrieve_kwargs: Keyword arguments to pass to the retrieval function
                             for this specific evaluation run.
                             
        Returns:
            A tuple of (EvaluationMetrics, detailed_results_list)
        """

        _kwargs = retrieve_kwargs if retrieve_kwargs else {}
        
        qa_pairs_to_run = self.qa_pairs
        
        if sample_size and sample_size < len(self.qa_pairs):
            if sample_indices_path and os.path.exists(sample_indices_path):
                with open(sample_indices_path, 'r') as f:
                    sample_indices = json.load(f)
                qa_pairs_to_run = [self.qa_pairs[i] for i in sample_indices if i < len(self.qa_pairs)]
                print(f"Loaded {len(qa_pairs_to_run)} samples from {sample_indices_path}")
            else:
                sample_indices = random.sample(range(len(self.qa_pairs)), sample_size)
                qa_pairs_to_run = [self.qa_pairs[i] for i in sample_indices]
                print(f"Generated {sample_size} random samples")

                if sample_indices_path:
                    os.makedirs(os.path.dirname(sample_indices_path), exist_ok=True)
                    with open(sample_indices_path, 'w') as f:
                        json.dump(sample_indices, f)
                    print(f"Saved sample indices to {sample_indices_path}")
        
        detailed_results = []
        recall_scores = {k: [] for k in self.k_values}
        precision_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        hit_rate_scores = {k: [] for k in self.k_values}
        rr_scores = []
        ap_scores = []

        print(f"Evaluating {len(qa_pairs_to_run)} queries for '{self.name}'...")
        if _kwargs:
            print(f"Using parameters: {_kwargs}")
            
        for qa_pair in tqdm(qa_pairs_to_run):
            result = self.evaluate_single_query(qa_pair, retrieve_kwargs=_kwargs)
            if result is None:
                continue
                
            detailed_results.append(result)

            for k in self.k_values:
                recall_scores[k].append(result['metrics'][f'recall@{k}'])
                precision_scores[k].append(result['metrics'][f'precision@{k}'])
                ndcg_scores[k].append(result['metrics'][f'ndcg@{k}'])
                hit_rate_scores[k].append(result['metrics'][f'hit_rate@{k}'])

            rr_scores.append(result['metrics']['reciprocal_rank'])
            ap_scores.append(result['metrics']['average_precision'])

        # Store results in the instance
        self.metrics = EvaluationMetrics(
            recall_at_k={k: np.mean(recall_scores[k]) for k in self.k_values},
            precision_at_k={k: np.mean(precision_scores[k]) for k in self.k_values},
            mrr=np.mean(rr_scores),
            map_score=np.mean(ap_scores),
            ndcg_at_k={k: np.mean(ndcg_scores[k]) for k in self.k_values},
            hit_rate_at_k={k: np.mean(hit_rate_scores[k]) for k in self.k_values}
        )
        self.detailed_results = detailed_results

        return self.metrics, self.detailed_results

    # --- NEW: Parameter Search ---

    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Helper to create all combinations from a parameter grid."""
        keys, values = param_grid.items()
        combinations = list(itertools.product(*values))
        param_sets = [dict(zip(keys, combo)) for combo in combinations]
        return param_sets

    def _get_optimization_score(self, metrics: EvaluationMetrics, metric_names: Union[str, List[str]]) -> float:
        """Helper to extract and average the target metric(s) from a metrics object."""
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        
        total_score = 0.0
        count = 0
        
        for name in metric_names:
            try:
                if name == 'mrr':
                    total_score += metrics.mrr
                    count += 1
                elif name == 'map_score':
                    total_score += metrics.map_score
                    count += 1
                elif name.startswith('recall@'):
                    k = int(name.split('@')[1])
                    total_score += metrics.recall_at_k[k]
                    count += 1
                elif name.startswith('precision@'):
                    k = int(name.split('@')[1])
                    total_score += metrics.precision_at_k[k]
                    count += 1
                elif name.startswith('ndcg@'):
                    k = int(name.split('@')[1])
                    total_score += metrics.ndcg_at_k[k]
                    count += 1
                elif name.startswith('hit_rate@'):
                    k = int(name.split('@')[1])
                    total_score += metrics.hit_rate_at_k[k]
                    count += 1
                else:
                    print(f"Warning: Unknown metric_name '{name}'. Skipping.")
            except KeyError:
                print(f"Warning: Metric '{name}' not found in results (k={k} not in k_values?). Skipping.")
            except Exception as e:
                print(f"Warning: Error processing metric '{name}': {e}. Skipping.")

        if count == 0:
            print("Error: No valid optimization metrics were found or calculated.")
            return 0.0
            
        return total_score / count

    def search_parameters(
        self,
        param_grid: Dict[str, List],
        metric_to_optimize: Union[str, List[str]],
        sample_size: int = None,
        sample_indices_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation over a grid of parameters and find the best set.

        Args:
            param_grid: A dictionary mapping parameter names to lists of values.
                        Example: {'weight': [0.1, 0.5], 'boost': [2, 5]}
            metric_to_optimize: The metric to maximize.
                                Examples: 'mrr', 'recall@5', 'ndcg@10'
                                Can also be a list to average: ['mrr', 'ndcg@10']
            sample_size: Number of queries to sample for this search.
                         (Recommended for speed).
            sample_indices_path: Path to load/save sample indices for consistent
                                 runs.
        
        Returns:
            A list of result dictionaries, sorted from best to worst score.
            Each dict contains: {'params': {...}, 'score': float, 'metrics': EvaluationMetrics}
        """
        print("Starting parameter search...")
        param_sets = self._generate_param_combinations(param_grid)
        print(f"Testing {len(param_sets)} parameter combinations.")
        
        search_results = []

        for i, params in enumerate(param_sets):
            print(f"\n--- Run {i+1}/{len(param_sets)} ---")
            
            # Run the full evaluation with this set of parameters
            metrics, _ = self.evaluate_all(
                sample_size=sample_size,
                sample_indices_path=sample_indices_path,
                retrieve_kwargs=params
            )
            
            # Get the score for this run
            score = self._get_optimization_score(metrics, metric_to_optimize)
            
            print(f"Parameters: {params}")
            print(f"Optimization Score ({metric_to_optimize}): {score:.4f}")
            
            search_results.append({
                'params': params,
                'score': score,
                'metrics': metrics  # The full EvaluationMetrics object
            })

        # Sort by score, descending
        search_results.sort(key=lambda x: x['score'], reverse=True)
        
        print("\n--- Parameter Search Complete ---")
        print(f"Best parameters found ( optimizing for '{metric_to_optimize}' ):")
        pprint(search_results[0])

        return search_results

    # --- Reporting and Saving (from helper functions) ---

    def plot_metrics(self, save_path: str = None):
        """
        Visualize evaluation metrics stored in self.metrics.
        
        Args:
            save_path: If provided, save the plot to this file path.
        """
        if not self.metrics:
            print(f"Error for '{self.name}': No metrics found. Run 'evaluate_all()' first.")
            return

        metrics = self.metrics  # Use the stored metrics

        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': '#f0f0f0',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#cccccc',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Retrieval Metrics: {self.approach_desc}\nModel: {self.model_name}', 
                     fontsize=20, y=1.03, fontweight='bold')
        fig.patch.set_facecolor('white')
        for ax in axes.flat:
            ax.set_facecolor('#f8f9fa')

        line_width = 3
        marker_size = 12
        font_size = 14
        title_size = 18
        k_vals = sorted(metrics.recall_at_k.keys())

        # Recall
        recall_vals = [metrics.recall_at_k[k] for k in k_vals]
        axes[0, 0].plot(k_vals, recall_vals, marker='o', linewidth=line_width,
                    markersize=marker_size, color='#2E86C1', label='Recall')
        axes[0, 0].set_xlabel('k', fontsize=font_size, fontweight='bold')
        axes[0, 0].set_ylabel('Recall@k', fontsize=font_size, fontweight='bold')
        axes[0, 0].set_title('Recall@k', fontsize=title_size, pad=20, fontweight='bold')
        axes[0, 0].set_ylim(max(0, min(recall_vals) - 0.1), min(1.0, max(recall_vals) + 0.1))
        axes[0, 0].legend(fontsize=font_size, loc='lower right')

        # Precision
        precision_vals = [metrics.precision_at_k[k] for k in k_vals]
        axes[0, 1].plot(k_vals, precision_vals, marker='s', linewidth=line_width,
                    markersize=marker_size, color='#E67E22', label='Precision')
        axes[0, 1].set_xlabel('k', fontsize=font_size, fontweight='bold')
        axes[0, 1].set_ylabel('Precision@k', fontsize=font_size, fontweight='bold')
        axes[0, 1].set_title('Precision@k', fontsize=title_size, pad=20, fontweight='bold')
        axes[0, 1].set_ylim(max(0, min(precision_vals) - 0.1), min(1.0, max(precision_vals) + 0.1))
        axes[0, 1].legend(fontsize=font_size, loc='upper right')

        # NDCG
        ndcg_vals = [metrics.ndcg_at_k[k] for k in k_vals]
        axes[1, 0].plot(k_vals, ndcg_vals, marker='^', linewidth=line_width,
                    markersize=marker_size, color='#27AE60', label='NDCG')
        axes[1, 0].set_xlabel('k', fontsize=font_size, fontweight='bold')
        axes[1, 0].set_ylabel('NDCG@k', fontsize=font_size, fontweight='bold')
        axes[1, 0].set_title('NDCG@k', fontsize=title_size, pad=20, fontweight='bold')
        axes[1, 0].set_ylim(max(0, min(ndcg_vals) - 0.1), min(1.0, max(ndcg_vals) + 0.1))
        axes[1, 0].legend(fontsize=font_size, loc='lower right')

        # Hit Rate
        hit_rate_vals = [metrics.hit_rate_at_k[k] for k in k_vals]
        axes[1, 1].plot(k_vals, hit_rate_vals, marker='d', linewidth=line_width,
                    markersize=marker_size, color='#C0392B', label='Hit Rate')
        axes[1, 1].set_xlabel('k', fontsize=font_size, fontweight='bold')
        axes[1, 1].set_ylabel('Hit Rate@k', fontsize=font_size, fontweight='bold')
        axes[1, 1].set_title('Hit Rate@k', fontsize=title_size, pad=20, fontweight='bold')
        axes[1, 1].set_ylim(max(0, min(hit_rate_vals) - 0.1), min(1.0, max(hit_rate_vals) + 0.1))
        axes[1, 1].legend(fontsize=font_size, loc='lower right')

        plt.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust layout for suptitle and summary

        # Summary text
        summary = f'Summary Metrics:\nMRR: {metrics.mrr:.4f}\nMAP: {metrics.map_score:.4f}'
        fig.text(0.99, 0.5, summary, fontsize=14, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Annotate points
        for ax in axes.flat:
            line = ax.get_lines()[0]
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
            plt.close()
        else:
            plt.show()

    def generate_report(self, metric_type: str = "retrieval") -> Dict:
        """
        Generate standardized report dictionary from self.metrics.
        
        Args:
            metric_type: A string to categorize this metric (e.g., "retrieval").
        
        Returns:
            A dictionary containing metrics and metadata.
        """
        if not self.metrics:
            raise ValueError("Metrics not calculated. Run evaluate_all() first.")
            
        return {
            'overall_metrics': {
                'recall_at_k': self.metrics.recall_at_k,
                'precision_at_k': self.metrics.precision_at_k,
                'ndcg_at_k': self.metrics.ndcg_at_k,
                'hit_rate_at_k': self.metrics.hit_rate_at_k,
                'mrr': self.metrics.mrr,
                'map': self.metrics.map_score
            },
            'metadata': {
                'name': self.name,
                'approach': self.approach_desc,
                'model': self.model_name,
                'metric_type': metric_type
            }
        }

    def save_results(
        self,
        evaluation_dir: str,
        metric_suffix: str = "",
        metric_type: str = "retrieval"
    ):
        """
        Save all evaluation artifacts (plot, report, details) for this approach.

        Args:
            evaluation_dir: The directory to save results to.
            metric_suffix: A suffix to append to filenames (e.g., "_m3").
        """
        if not self.metrics or not self.detailed_results:
            raise ValueError(f"Metrics for '{self.name}' not calculated. Run evaluate_all() first.")

        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Clean up suffix
        suffix = f"_{metric_suffix}" if metric_suffix else ""
        
        # Save visualization
        plot_path = f'{evaluation_dir}/evaluation_metrics_{self.name}{suffix}.png'
        self.plot_metrics(save_path=plot_path)
        
        # Save report
        report_path = f'{evaluation_dir}/evaluation_report_{self.name}{suffix}.json'
        report = self.generate_report(metric_type)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Save detailed results
        details_path = f'{evaluation_dir}/detailed_results_{self.name}{suffix}.json'
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_results, f, ensure_ascii=False, indent=2)
            
        print(f"✓ Saved all results for '{self.name}' to {evaluation_dir}")


# --- Standalone Comparison Functions ---
# (These operate on a LIST of RetrievalEvaluator objects)

def calc_improvement(original: float, new: float) -> float:
    """Calculate percentage improvement with safe division."""
    if original == 0 and new > 0:
        return float('inf') # Or 100.0 if you prefer
    if original == 0 and new == 0:
        return 0.0
    return ((new - original) / original) * 100

def print_comparison_details(
    baseline_val: float, 
    new_val: float, 
    k: str, 
    metric_name: str,
    approach_desc: str
):
    """Print improvement details for a specific metric comparison."""
    imp = calc_improvement(baseline_val, new_val)
    sign = "+" if imp >= 0 else ""
    print(f"      {approach_desc:<25} vs Baseline: {sign}{imp:,.2f}% ({metric_name}@{k})")

def compare_approaches(
    approaches: List[RetrievalEvaluator],
    metric_type: str,
    comparison_func: Callable = None
):
    """
    Generate a formatted markdown comparison table between all approaches.

    Args:
        approaches: A list of *evaluated* RetrievalEvaluator objects.
        metric_type: A string label for the comparison.
        comparison_func: Optional function to print detailed diffs
                         (e.g., print_comparison_details).
    """
    if len(approaches) < 1:
        print("No approaches to compare.")
        return
        
    # Check if evaluators have been run
    for a in approaches:
        if not a.metrics:
            print(f"Warning: Approach '{a.name}' has no metrics. Skipping.")
            approaches.remove(a)
            
    if len(approaches) < 1:
        print("No *evaluated* approaches to compare.")
        return
        
    # Identify baseline (first approach marked as baseline, or just first)
    baseline = next((a for a in approaches if a.is_baseline), approaches[0])
    comparisons = [a for a in approaches if a != baseline]
    
    all_approaches = [baseline] + comparisons
    
    # Create header with full approach names
    headers = ["Metric"] + [a.approach_desc for a in all_approaches]
    
    # Build metric rows
    rows = []
    
    # Get all K values
    all_ks = sorted(set(
        k for a in all_approaches 
        for k in a.metrics.recall_at_k.keys()
    ))
    
    # --- K-Metrics Section ---
    k_metrics = [
        ("Recall", lambda a, k: a.metrics.recall_at_k.get(k, 0.0)),
        ("Precision", lambda a, k: a.metrics.precision_at_k.get(k, 0.0)),
        ("NDCG", lambda a, k: a.metrics.ndcg_at_k.get(k, 0.0)),
        ("Hit Rate", lambda a, k: a.metrics.hit_rate_at_k.get(k, 0.0)),
    ]
    
    for metric_name, metric_func in k_metrics:
        for k in all_ks:
            metric_label = f"{metric_name}@{k}"
            row = [metric_label]
            baseline_val = metric_func(baseline, k)
            
            for approach in all_approaches:
                val = metric_func(approach, k)
                if approach == baseline:
                    row.append(f"{val:.4f}")
                else:
                    imp = calc_improvement(baseline_val, val)
                    sign = "+" if imp >= 0 else ""
                    row.append(f"{val:.4f} ({sign}{imp:,.1f}%)")
            rows.append(row)

    # --- Single-Value Metrics Section ---
    single_metrics = [
        ("MRR", lambda a: a.metrics.mrr),
        ("MAP", lambda a: a.metrics.map_score)
    ]
    
    for metric_name, metric_func in single_metrics:
        row = [metric_name]
        baseline_val = metric_func(baseline)
        
        for approach in all_approaches:
            val = metric_func(approach)
            if approach == baseline:
                row.append(f"{val:.4f}")
            else:
                imp = calc_improvement(baseline_val, val)
                sign = "+" if imp >= 0 else ""
                row.append(f"{val:.4f} ({sign}{imp:,.1f}%)")
        rows.append(row)

    # Generate markdown table
    print(f"\n## METRIC COMPARISON: {metric_type.upper()} | BASELINE: {baseline.approach_desc}\n")
    
    # Calculate column widths (minimum 8 for metric names)
    col_widths = [max(len(str(cell)) for cell in [header] + [row[i] for row in rows]) 
                 for i, header in enumerate(headers)]
    
    # Header row
    header_row = "| " + " | ".join(
        f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))
    ) + " |"
    
    # Separator row
    sep_row = "| " + " | ".join(
        "-" * (col_widths[i] + 1) for i in range(len(headers))
    ) + " |"
    
    # Data rows
    data_rows = []
    for row in rows:
        data_rows.append("| " + " | ".join(
            f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)
        ) + " |")
    
    # Print the table
    print(header_row)
    print(sep_row)
    for row in data_rows:
        print(row)
    
    print("\n*Improvement is calculated against baseline. Positive values indicate improvement.*")
