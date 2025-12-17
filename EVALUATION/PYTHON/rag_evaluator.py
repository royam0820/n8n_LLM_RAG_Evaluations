"""
RAG System Evaluation Tool
Supports multiple evaluation methods: LangChain, Gemini, and custom metrics
"""

import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from dataclasses import dataclass, asdict

@dataclass
class EvaluationResult:
    question: str
    reference_answer: str
    actual_answer: str
    correctness_score: float
    reasoning: str
    execution_time: float
    token_count: int
    estimated_cost: float
    timestamp: str

class RAGEvaluator:
    def __init__(self, method: str = "gemini", gemini_api_key: Optional[str] = None):
        """
        Initialize the evaluator
        
        Args:
            method: "gemini", "langchain", or "both"
            gemini_api_key: Your Gemini API key (if using gemini method)
        """
        self.method = method
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if method in ["gemini", "both"]:
            self._init_gemini()
        
        if method in ["langchain", "both"]:
            self._init_langchain()
    
    def _init_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("✓ Gemini initialized")
        except ImportError:
            print("⚠ google-generativeai not installed. Run: pip install google-generativeai")
            self.gemini_model = None
        except Exception as e:
            print(f"⚠ Gemini initialization failed: {e}")
            self.gemini_model = None
    
    def _init_langchain(self):
        """Initialize LangChain evaluator"""
        try:
            from langchain.evaluation import load_evaluator
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            self.correctness_evaluator = load_evaluator(
                "labeled_criteria",
                criteria="correctness",
                llm=self.llm
            )
            print("✓ LangChain evaluator initialized")
        except ImportError:
            print("⚠ LangChain not installed. Run: pip install langchain langchain-openai")
            self.correctness_evaluator = None
        except Exception as e:
            print(f"⚠ LangChain initialization failed: {e}")
            self.correctness_evaluator = None
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def _evaluate_with_gemini(self, question: str, reference: str, actual: str) -> Dict:
        """Evaluate using Gemini"""
        if not self.gemini_model:
            return {"score": 0, "reasoning": "Gemini not available"}
        
        evaluation_prompt = f"""You are an expert evaluator for RAG systems. Evaluate the quality of the answer provided.

Question: {question}

Reference Answer (ideal): {reference}

Actual Answer (from system): {actual}

Evaluate the actual answer against the reference answer based on:
1. Correctness - Is the information accurate?
2. Completeness - Does it cover all key points?
3. Relevance - Does it answer the question directly?

Provide a score from 0 to 1 (where 1 is perfect) and explain your reasoning.

Return your evaluation as JSON in this exact format:
{{"score": 0.0, "reasoning": "explanation here"}}

Only return the JSON, nothing else."""

        try:
            response = self.gemini_model.generate_content(evaluation_prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result
        except Exception as e:
            print(f"⚠ Gemini evaluation error: {e}")
            return {"score": 0, "reasoning": f"Evaluation failed: {str(e)}"}
    
    def _evaluate_with_langchain(self, question: str, reference: str, actual: str) -> Dict:
        """Evaluate using LangChain"""
        if not self.correctness_evaluator:
            return {"score": 0, "reasoning": "LangChain not available"}
        
        try:
            result = self.correctness_evaluator.evaluate_strings(
                prediction=actual,
                reference=reference,
                input=question
            )
            return {
                "score": result.get('score', 0),
                "reasoning": result.get('reasoning', 'No reasoning provided')
            }
        except Exception as e:
            print(f"⚠ LangChain evaluation error: {e}")
            return {"score": 0, "reasoning": f"Evaluation failed: {str(e)}"}
    
    def evaluate_single(self, question: str, reference_answer: str, 
                       actual_answer: str) -> EvaluationResult:
        """Evaluate a single Q&A pair"""
        start_time = time.time()
        
        # Choose evaluation method
        if self.method == "gemini":
            eval_result = self._evaluate_with_gemini(question, reference_answer, actual_answer)
        elif self.method == "langchain":
            eval_result = self._evaluate_with_langchain(question, reference_answer, actual_answer)
        else:  # both - use Gemini as primary
            eval_result = self._evaluate_with_gemini(question, reference_answer, actual_answer)
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        total_tokens = self._estimate_tokens(question + reference_answer + actual_answer)
        # Rough cost estimate (Gemini Flash is very cheap, ~$0.00001 per 1K tokens)
        estimated_cost = (total_tokens / 1000) * 0.00001
        
        return EvaluationResult(
            question=question,
            reference_answer=reference_answer,
            actual_answer=actual_answer,
            correctness_score=eval_result['score'],
            reasoning=eval_result['reasoning'],
            execution_time=execution_time,
            token_count=total_tokens,
            estimated_cost=estimated_cost,
            timestamp=datetime.now().isoformat()
        )
    
    def evaluate_dataset(self, data: pd.DataFrame, 
                        question_col: str = "question",
                        reference_col: str = "reference_answer", 
                        actual_col: str = "actual_answer") -> pd.DataFrame:
        """
        Evaluate an entire dataset
        
        Args:
            data: DataFrame with questions, reference answers, and actual answers
            question_col: Name of the question column
            reference_col: Name of the reference answer column
            actual_col: Name of the actual answer column
        
        Returns:
            DataFrame with evaluation results
        """
        print(f"\n🔍 Starting evaluation of {len(data)} items using {self.method}...\n")
        
        results = []
        for idx, row in data.iterrows():
            print(f"Evaluating item {idx + 1}/{len(data)}...", end=" ")
            
            result = self.evaluate_single(
                question=row[question_col],
                reference_answer=row[reference_col],
                actual_answer=row[actual_col]
            )
            
            results.append(asdict(result))
            print(f"✓ Score: {result.correctness_score:.2f}")
        
        results_df = pd.DataFrame(results)
        
        # Print summary statistics
        self._print_summary(results_df)
        
        return results_df
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"Total evaluations: {len(results_df)}")
        print(f"Average correctness score: {results_df['correctness_score'].mean():.3f}")
        print(f"Median correctness score: {results_df['correctness_score'].median():.3f}")
        print(f"Min score: {results_df['correctness_score'].min():.3f}")
        print(f"Max score: {results_df['correctness_score'].max():.3f}")
        print(f"\nTotal tokens processed: {results_df['token_count'].sum():,}")
        print(f"Average tokens per evaluation: {results_df['token_count'].mean():.0f}")
        print(f"Total execution time: {results_df['execution_time'].sum():.2f}s")
        print(f"Average execution time: {results_df['execution_time'].mean():.2f}s")
        print(f"Estimated total cost: ${results_df['estimated_cost'].sum():.6f}")
        print("="*60 + "\n")
    
    def save_results(self, results_df: pd.DataFrame, output_path: str = "evaluation_results.csv"):
        """Save results to CSV"""
        results_df.to_csv(output_path, index=False)
        print(f"✓ Results saved to {output_path}")
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str = "evaluation_report.html"):
        """Generate an HTML report"""
        # Calculate score distribution
        score_bins = pd.cut(results_df['correctness_score'], 
                           bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
        score_dist = score_bins.value_counts().sort_index()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 24px; color: #4CAF50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .score-good {{ color: #4CAF50; font-weight: bold; }}
        .score-medium {{ color: #FF9800; font-weight: bold; }}
        .score-bad {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 RAG System Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <div class="metric-label">Total Evaluations</div>
                <div class="metric-value">{len(results_df)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Average Score</div>
                <div class="metric-value">{results_df['correctness_score'].mean():.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Tokens</div>
                <div class="metric-value">{results_df['token_count'].sum():,}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Time</div>
                <div class="metric-value">{results_df['execution_time'].sum():.2f}s</div>
            </div>
        </div>
        
        <h2>Score Distribution</h2>
        <table>
            <tr><th>Score Range</th><th>Count</th><th>Percentage</th></tr>
"""
        
        for score_range, count in score_dist.items():
            pct = (count / len(results_df)) * 100
            html += f"            <tr><td>{score_range}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"
        
        html += """
        </table>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Question</th>
                <th>Score</th>
                <th>Tokens</th>
                <th>Time (s)</th>
                <th>Reasoning</th>
            </tr>
"""
        
        for _, row in results_df.iterrows():
            score_class = 'score-good' if row['correctness_score'] >= 0.7 else ('score-medium' if row['correctness_score'] >= 0.4 else 'score-bad')
            html += f"""
            <tr>
                <td>{row['question'][:100]}...</td>
                <td class="{score_class}">{row['correctness_score']:.2f}</td>
                <td>{row['token_count']}</td>
                <td>{row['execution_time']:.2f}</td>
                <td>{row['reasoning'][:150]}...</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Report generated: {output_path}")


# Example usage function
def main():
    """Example usage of the RAG Evaluator"""
    
    # Example 1: Create sample data
    sample_data = pd.DataFrame([
        {
            "question": "What is the capital of France?",
            "reference_answer": "The capital of France is Paris.",
            "actual_answer": "Paris is the capital city of France."
        },
        {
            "question": "What is photosynthesis?",
            "reference_answer": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
            "actual_answer": "It's when plants make food using sunlight."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "reference_answer": "William Shakespeare wrote Romeo and Juliet.",
            "actual_answer": "Shakespeare wrote it."
        }
    ])
    
    # Initialize evaluator (using Gemini - it's free and works well)
    evaluator = RAGEvaluator(method="gemini")
    
    # Evaluate the dataset
    results = evaluator.evaluate_dataset(sample_data)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.csv")
    
    # Generate HTML report
    evaluator.generate_report(results, "evaluation_report.html")
    
    print("\n✓ Evaluation complete!")
    print("  - Results saved to: evaluation_results.csv")
    print("  - Report saved to: evaluation_report.html")


if __name__ == "__main__":
    main()
