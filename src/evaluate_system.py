"""
Evaluation Framework for PaperMind
Measures retrieval accuracy, answer quality, and performance
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
from multimodal_rag_pipeline import MultiModalRAG
from datetime import datetime


class SystemEvaluator:
    """Evaluate RAG system performance"""

    def __init__(self):
        print("Initializing evaluator...")
        self.rag = MultiModalRAG()
        self.results = []

    def load_test_queries(self, path: str = "data/evaluation/test_queries.json") -> List[Dict]:
        """Load test query dataset"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data['queries']

    def evaluate_retrieval(self, query: Dict, text_k: int = 5) -> Dict:
        """
        Evaluate retrieval quality for a single query

        Metrics:
        - Retrieval precision: % of expected papers in top-k
        - Mean relevance score
        - Retrieval latency
        """
        start_time = time.time()

        # Retrieve
        text_results, image_results = self.rag.retrieve_multimodal(
            query['question'],
            text_k=text_k,
            image_k=2
        )

        retrieval_time = time.time() - start_time

        # Check if expected papers were retrieved
        retrieved_papers = [r['paper'] for r in text_results]
        expected_papers = query.get('expected_papers', [])

        # Calculate precision (how many expected papers in top-k)
        if expected_papers:
            matches = sum(1 for paper in retrieved_papers
                          if any(exp in paper for exp in expected_papers))
            precision = matches / len(expected_papers)
        else:
            precision = None

        # Mean relevance score
        relevance_scores = [r['score'] for r in text_results]
        mean_score = np.mean(relevance_scores)

        return {
            'query_id': query['id'],
            'question': query['question'],
            'category': query['category'],
            'difficulty': query['difficulty'],
            'retrieval_time': retrieval_time,
            'top_relevance': relevance_scores[0],
            'mean_relevance': mean_score,
            'precision': precision,
            'retrieved_papers': retrieved_papers[:3],
            'expected_papers': expected_papers
        }

    def evaluate_end_to_end(self, query: Dict) -> Dict:
        """
        Evaluate complete end-to-end query

        Metrics:
        - Total latency
        - Token usage
        - Answer length
        - Citation count
        """
        start_time = time.time()

        result = self.rag.query(
            query['question'],
            text_k=3,
            image_k=2,
            verbose=False
        )

        total_time = time.time() - start_time

        # Count citations in answer
        answer = result['answer']
        text_citations = answer.count('[Text Source')
        figure_citations = answer.count('[Figure')

        return {
            'query_id': query['id'],
            'question': query['question'],
            'total_latency': total_time,
            'retrieval_latency': total_time - result['latency'],
            'llm_latency': result['latency'],
            'tokens_used': result['tokens_used'],
            'answer_length': len(answer),
            'text_citations': text_citations,
            'figure_citations': figure_citations,
            'answer': answer
        }

    def run_evaluation(self, test_queries: List[Dict]) -> Dict:
        """Run complete evaluation suite"""

        print("-" * 70)
        print("RUNNING SYSTEM EVALUATION")
        print("-" * 70)
        print(f"Test queries: {len(test_queries)}\n")

        retrieval_results = []
        e2e_results = []

        for i, query in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] Evaluating: {query['question'][:50]}...")

            # Retrieval evaluation
            ret_result = self.evaluate_retrieval(query)
            retrieval_results.append(ret_result)

            # End-to-end evaluation
            e2e_result = self.evaluate_end_to_end(query)
            e2e_results.append(e2e_result)

            time.sleep(0.5)  # Rate limiting

        # Calculate aggregate metrics
        aggregate = self.calculate_aggregate_metrics(retrieval_results, e2e_results)

        return {
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(test_queries),
            'retrieval_results': retrieval_results,
            'e2e_results': e2e_results,
            'aggregate_metrics': aggregate
        }

    def calculate_aggregate_metrics(self, retrieval_results: List[Dict], e2e_results: List[Dict]) -> Dict:
        """Calculate aggregate performance metrics"""

        # Retrieval metrics
        precisions = [r['precision'] for r in retrieval_results if r['precision'] is not None]
        mean_precision = np.mean(precisions) if precisions else 0

        top_relevances = [r['top_relevance'] for r in retrieval_results]
        mean_relevances = [r['mean_relevance'] for r in retrieval_results]
        retrieval_times = [r['retrieval_time'] for r in retrieval_results]

        # E2E metrics
        total_latencies = [r['total_latency'] for r in e2e_results]
        llm_latencies = [r['llm_latency'] for r in e2e_results]
        tokens = [r['tokens_used'] for r in e2e_results]
        answer_lengths = [r['answer_length'] for r in e2e_results]
        text_cites = [r['text_citations'] for r in e2e_results]
        fig_cites = [r['figure_citations'] for r in e2e_results]

        return {
            'retrieval': {
                'mean_precision': mean_precision,
                'mean_top_relevance': np.mean(top_relevances),
                'mean_avg_relevance': np.mean(mean_relevances),
                'mean_retrieval_time': np.mean(retrieval_times),
                'p50_retrieval_time': np.percentile(retrieval_times, 50),
                'p95_retrieval_time': np.percentile(retrieval_times, 95)
            },
            'end_to_end': {
                'mean_total_latency': np.mean(total_latencies),
                'p50_total_latency': np.percentile(total_latencies, 50),
                'p95_total_latency': np.percentile(total_latencies, 95),
                'mean_llm_latency': np.mean(llm_latencies),
                'mean_tokens': np.mean(tokens),
                'mean_answer_length': np.mean(answer_lengths),
                'mean_text_citations': np.mean(text_cites),
                'mean_figure_citations': np.mean(fig_cites)
            },
            'cost': {
                'avg_cost_per_query': np.mean(tokens) * 0.002 / 1000,  # GPT-3.5 pricing
                'total_eval_cost': sum(tokens) * 0.002 / 1000
            }
        }

    def generate_report(self, results: Dict, output_path: str = "data/evaluation/results.json"):
        """Generate evaluation report"""

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Convert results
        results_native = convert_to_native(results)

        # Save detailed results
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(results_native, f, indent=2)

        # Print summary
        metrics = results['aggregate_metrics']

        print("\n" + "-" * 70)
        print("EVALUATION RESULTS")
        print("-" * 70)

        print("\nRetrieval Performance:")
        print(f"  Mean Precision: {metrics['retrieval']['mean_precision']:.2%}")
        print(f"  Top Relevance Score: {metrics['retrieval']['mean_top_relevance']:.3f}")
        print(f"  Avg Relevance Score: {metrics['retrieval']['mean_avg_relevance']:.3f}")
        print(f"  Retrieval Time (P50): {metrics['retrieval']['p50_retrieval_time'] * 1000:.1f}ms")

        print("\nEnd-to-End Performance:")
        print(f"  Total Latency (P50): {metrics['end_to_end']['p50_total_latency']:.2f}s")
        print(f"  LLM Latency: {metrics['end_to_end']['mean_llm_latency']:.2f}s")
        print(f"  Avg Tokens per Query: {metrics['end_to_end']['mean_tokens']:.0f}")
        print(f"  Avg Answer Length: {metrics['end_to_end']['mean_answer_length']:.0f} chars")
        print(f"  Avg Text Citations: {metrics['end_to_end']['mean_text_citations']:.1f}")
        print(f"  Avg Figure Citations: {metrics['end_to_end']['mean_figure_citations']:.1f}")

        print("\nCost Analysis:")
        print(f"  Cost per Query: ${metrics['cost']['avg_cost_per_query']:.4f}")
        print(f"  Total Evaluation Cost: ${metrics['cost']['total_eval_cost']:.4f}")

        print(f"\nDetailed results saved to: {output_path}")
        print("-" * 70)


if __name__ == "__main__":
    evaluator = SystemEvaluator()

    # Load test queries
    queries = evaluator.load_test_queries()

    # Run evaluation
    results = evaluator.run_evaluation(queries)

    # Generate report
    evaluator.generate_report(results)