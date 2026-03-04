"""
RAG GENERATION EVALUATION SYSTEM
- Uses Ollama models + Gemini 2.0 Flash Lite for generation
- Uses Ollama (bge-m3) for Embeddings
- Evaluates using GPT-4o-mini as judge
- Metrics: Semantic Similarity, Faithfulness, Completeness, Overall Score
"""

import os
import json
import time
import requests
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Removed SentenceTransformer import
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
try:
    from Evaluate_Retrieval_With_Reranker_Template import HybridRetriever
except ImportError:
    raise ImportError("Could not import HybridRetriever. Make sure 'Evaluate_Retrieval_With_Reranker.py' is in the same directory.")

# ==================================================
# CONFIGURATION
# ==================================================

# Paths
EVALUATION_QUESTIONS_PATH = "/home/olj3kor/praveen/RAG_work/evaluation_questions.json"
COLLECTION = "rag_hybrid_bge_m3"
QDRANT_URL = "http://localhost:7333"

# Embedding Config
EMBEDDING_MODEL = "bge-m3"  # Name of the model in Ollama

# Output paths
GENERATION_RESULTS_PATH = "/home/olj3kor/praveen/RAG_work/generation_results.json"
EVALUATION_RESULTS_PATH = "/home/olj3kor/praveen/RAG_work/generation_evaluation.json"

# Retrieval config
TOP_K = 10  # Number of documents to retrieve

# API Keys
GPT4O_MINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Uses same key
if not GPT4O_MINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

# Endpoints
GPT4O_MINI_ENDPOINT = "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview"
OLLAMA_BASE_URL = "http://localhost:11434"

# ==================================================
# OLLAMA INTEGRATION (GENERATION & EMBEDDING)
# ==================================================

def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        return []
    except Exception as e:
        print(f"Warning: Could not connect to Ollama: {e}")
        return []

def get_ollama_embedding(text: str, model_name: str) -> List[float]:
    """
    Generate embedding using Ollama API
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    
    payload = {
        "model": model_name,
        "prompt": text
    }
    
    try:
        # Increased timeout for embeddings
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Ollama Embedding Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error generating embedding with {model_name}: {e}")
        # Return empty list or handle appropriately - here we raise to stop execution if retrieval fails
        raise e

def generate_with_ollama(model_name: str, prompt: str, temperature: float = 0.1) -> str:
    """Generate response using Ollama model"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
        "options": {
            "num_predict": 512,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


# ==================================================
# GEMINI INTEGRATION
# ==================================================

def generate_with_gemini(prompt: str, temperature: float = 0.1) -> str:
    """Generate response using Gemini 2.0 Flash Lite via Bosch endpoint"""
    headers = {
        "genaiplatform-farm-subscription-key": GPT4O_MINI_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(
            GPT4O_MINI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


# ==================================================
# GPT-4O-MINI AS JUDGE
# ==================================================

def evaluate_with_gpt4o_mini(question: str, context: str, answer: str, 
                             ground_truth: str) -> Dict[str, Any]:
    """
    Use GPT-4o-mini to evaluate answer quality
    Returns: faithfulness score, completeness score, explanation
    """
    
    judge_prompt = f"""You are a QA Auditor for an Automotive Technical Support System.
        Your job is to evaluate if the AI-generated answer is safe, accurate, and fully supported by the reference text.

        ### INPUTS
        **User Question:** {question}
        **Retrieved Context:** {context[:3000]}
        **Generated Answer:** {answer}
        **Ground Truth:** {ground_truth}

        ### EVALUATION CRITERIA

        1. Faithfulness (Score 0-10):
        - 10: Every claim is directly supported by the Context.
        - 0: The answer contains information NOT present in the Context (Hallucination), even if factually correct in the real world. Strict penalty for external knowledge.

        2. Completeness (Score 0-10):
        - 10: Fully answers the question and includes all relevant details from the Context.
        - 0: Misses key details explicitly stated in the Context.

        3. Correctness (Score 0-10):
        - 10: All technical details (numbers, names, renamings, API additions, terminology, warnings) exactly match the Context and Ground Truth.
        - 5: Partially correct; minor inaccuracies or vague phrasing.
        - 0: Incorrect, misleading, or omits critical technical details.

        Return ONLY a JSON object with this exact format:
        {{
        "faithfulness": <score 0-10>,
        "completeness": <score 0-10>,
        "correctness": <score 0-10>,
        "explanation": "<brief 1-2 sentence explanation>"
        }}"""

    headers = {
        "genaiplatform-farm-subscription-key": GPT4O_MINI_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {"role": "user", "content": judge_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }
    
    try:
        response = requests.post(
            GPT4O_MINI_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            
            # Clean and parse JSON
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            scores = json.loads(response_text)
            return scores
        else:
            return {
                "faithfulness": 0,
                "completeness": 0,
                "correctness": 0,
                "explanation": f"Judge API error: {response.status_code}"
            }
    except Exception as e:
        return {
            "faithfulness": 0,
            "completeness": 0,
            "correctness": 0,
            "explanation": f"Judge error: {str(e)}"
        }


# ==================================================
# RAG SYSTEM
# ==================================================

class RAGSystem:
    def __init__(self, qdrant_url: str, collection: str, embedding_model: str):
        print("Initializing Hybrid RAG System...")
        # Initialize the advanced retriever from your other script
        self.retriever = HybridRetriever(
            qdrant_url=qdrant_url,
            collection_name=collection,
            use_ollama=True,
            use_reranker=True
        )
        print("✓ Hybrid RAG System initialized")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict], str]:
        """Retrieve using HybridRetriever"""
        # 1. Call the advanced search
        results = self.retriever.search(query, top_k=top_k)
        
        # 2. Convert results to the format your generation code expects
        docs = []
        for res in results:
            docs.append({
                "content": res.content,
                "score": res.score,
                "source": res.metadata.get('filename', 'unknown')
            })
        
        # 3. Build context string
        context = "\n\n".join([
            f"[Document: {d['source']}] (Score: {d['score']:.4f})\n{d['content']}" 
            for d in docs
        ])
        
        return docs, context

    def build_prompt(self, question: str, context: str) -> str:
        """
        Builds a production-grade RAG prompt with strict constraints.
        """
        prompt = f"""### SYSTEM ROLE
            You are a Senior Automotive Technical Expert and Documentation Assistant. 
            Your task is to answer user questions strictly based on the provided technical documentation (Context).
            Your answers must be precise, safety-conscious, and suitable for professional engineers and technicians.

            ### STRICT OPERATING RULES
            1. **NO OUTSIDE KNOWLEDGE:** You must rely ONLY on the provided Context. Do not use external training data.
            2. **NO HALLUCINATION:** If the answer is not in the context, state: "The provided documents do not contain the specific information to answer this question." Do not guess torque specs, dimensions, or safety procedures.
            3. **CITATIONS:** Whenever possible, reference the specific section or document name from the context (e.g., "[Doc 1]", "Section 4.2").
            4. **SAFETY FIRST:** If the context mentions safety warnings (WARNING, CAUTION, DANGER), you must include them in your answer.
            5. **NEUTRAL TONE:** Maintain a professional, objective, and technical tone. Avoid conversational filler.

            ### INPUT DATA
            <context>
            {context}
            </context>

            <user_query>
            {question}
            </user_query>

            ### OUTPUT FORMAT
            Provide your response in the following structure:
            **Answer:** [Direct answer to the question]
            **Evidence:** [Quotes or references from the text supporting your answer]
            **Safety Notes:** [Any relevant warnings found in the text; if none, omit this]

            ### RESPONSE
            """
        return prompt


# ==================================================
# GENERATION PIPELINE
# ==================================================

def run_generation_pipeline(questions_data: Dict, rag_system: RAGSystem,
                            ollama_models: List[str]) -> Dict[str, Any]:
    """
    Run generation for all questions with all models
    """
    print("\n" + "="*80)
    print("STARTING GENERATION PIPELINE")
    print("="*80)
    
    questions = questions_data["questions"]
    total_models = len(ollama_models) + 1  # +1 for Gemini
    
    print(f"Questions: {len(questions)}")
    print(f"Ollama models: {len(ollama_models)}")
    print(f"+ Gemini 2.0 Flash Lite")
    print(f"Total generations: {len(questions) * total_models}")
    print("="*80)
    
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_questions": len(questions),
            "num_models": total_models,
            "models": ollama_models + ["gemini-2.0-flash-lite"],
            "top_k_retrieval": TOP_K
        },
        "generations": []
    }
    
    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {q_idx}/{len(questions)}: {question['question'][:60]}...")
        print(f"{'='*80}")
        
        # Retrieve context
        print("  → Retrieving context...")
        docs, context = rag_system.retrieve(question["question"], top_k=TOP_K)
        print(f"  ✓ Retrieved {len(docs)} documents")
        
        # Build prompt
        prompt = rag_system.build_prompt(question["question"], context)
        
        # Store generation for this question
        question_result = {
            "question_id": question["id"],
            "question": question["question"],
            "ground_truth_answer": question["answer"],
            "source_document": question["source_document"],
            "retrieved_docs": [{"source": d["source"], "score": d["score"]} for d in docs],
            "context": context[:1000] + "..." if len(context) > 1000 else context,
            "model_answers": {}
        }
        
        # Generate with Ollama models
        for model_name in ollama_models:
            print(f"  → Generating with {model_name}...")
            start_time = time.time()
            answer = generate_with_ollama(model_name, prompt)
            latency = time.time() - start_time
            
            question_result["model_answers"][model_name] = {
                "answer": answer,
                "latency_seconds": latency
            }
            print(f"    ✓ Generated ({latency:.2f}s)")
        
        # Generate with Gemini
        print(f"  → Generating with Gemini 2.0 Flash Lite...")
        start_time = time.time()
        answer = generate_with_gemini(prompt)
        latency = time.time() - start_time
        
        question_result["model_answers"]["gemini-2.0-flash-lite"] = {
            "answer": answer,
            "latency_seconds": latency
        }
        print(f"    ✓ Generated ({latency:.2f}s)")
        
        results["generations"].append(question_result)
    
    return results


# ==================================================
# EVALUATION PIPELINE
# ==================================================

def calculate_semantic_similarity(answer1: str, answer2: str, 
                                  embedding_model_name: str) -> float:
    """Calculate semantic similarity between two answers using Ollama Embeddings"""
    if not answer1.strip() or not answer2.strip():
        return 0.0
    
    # Get embeddings via Ollama
    emb1_list = get_ollama_embedding(answer1, embedding_model_name)
    emb2_list = get_ollama_embedding(answer2, embedding_model_name)
    
    # Convert to numpy array and reshape for sklearn
    emb1 = np.array(emb1_list).reshape(1, -1)
    emb2 = np.array(emb2_list).reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)


def run_evaluation_pipeline(generation_results: Dict, 
                            embedding_model_name: str) -> Dict[str, Any]:
    """
    Evaluate all generated answers using GPT-4o-mini as judge
    Compare all models against Gemini as baseline
    """
    print("\n" + "="*80)
    print("STARTING EVALUATION PIPELINE")
    print("="*80)
    print("Judge: GPT-4o-mini")
    print("Baseline: Gemini 2.0 Flash Lite")
    print(f"Similarity Model: {embedding_model_name} (Ollama)")
    print("="*80)
    
    evaluations = []
    model_scores = defaultdict(lambda: {
        "faithfulness": [],
        "completeness": [],
        "correctness": [],
        "semantic_similarity": [],
        "overall": []
    })
    
    generations = generation_results["generations"]
    
    for q_idx, gen in enumerate(generations, 1):
        print(f"\nEvaluating question {q_idx}/{len(generations)}...")
        
        question = gen["question"]
        ground_truth = gen["ground_truth_answer"]
        context = gen["context"]
        
        # Get Gemini answer as baseline
        gemini_answer = gen["model_answers"]["gemini-2.0-flash-lite"]["answer"]
        
        question_eval = {
            "question_id": gen["question_id"],
            "question": question,
            "model_evaluations": {}
        }
        
        # Evaluate each model
        for model_name, model_data in gen["model_answers"].items():
            answer = model_data["answer"]
            
            print(f"  → Evaluating {model_name}...")
            
            # Get LLM-as-judge scores
            judge_scores = evaluate_with_gpt4o_mini(
                question=question,
                context=context,
                answer=answer,
                ground_truth=ground_truth
            )
            
            # Calculate semantic similarity with Gemini
            if model_name != "gemini-2.0-flash-lite":
                sem_sim = calculate_semantic_similarity(
                    answer, gemini_answer, embedding_model_name
                )
            else:
                sem_sim = 1.0  # Gemini compared to itself
            
            # Calculate overall score (weighted average)
            overall_score = (
                judge_scores.get("faithfulness", 0) * 0.35 +
                judge_scores.get("completeness", 0) * 0.35 +
                judge_scores.get("correctness", 0) * 0.30
            )
            
            # Store evaluation
            question_eval["model_evaluations"][model_name] = {
                "faithfulness": judge_scores.get("faithfulness", 0),
                "completeness": judge_scores.get("completeness", 0),
                "correctness": judge_scores.get("correctness", 0),
                "semantic_similarity_to_gemini": sem_sim,
                "overall_score": overall_score,
                "judge_explanation": judge_scores.get("explanation", ""),
                "latency_seconds": model_data["latency_seconds"]
            }
            
            # Aggregate for model-level stats
            model_scores[model_name]["faithfulness"].append(
                judge_scores.get("faithfulness", 0)
            )
            model_scores[model_name]["completeness"].append(
                judge_scores.get("completeness", 0)
            )
            model_scores[model_name]["correctness"].append(
                judge_scores.get("correctness", 0)
            )
            model_scores[model_name]["semantic_similarity"].append(sem_sim)
            model_scores[model_name]["overall"].append(overall_score)
        
        evaluations.append(question_eval)
    
    # Calculate aggregate statistics
    aggregate_stats = {}
    for model_name, scores in model_scores.items():
        aggregate_stats[model_name] = {
            "mean_faithfulness": float(np.mean(scores["faithfulness"])),
            "mean_completeness": float(np.mean(scores["completeness"])),
            "mean_correctness": float(np.mean(scores["correctness"])),
            "mean_semantic_similarity_to_gemini": float(np.mean(scores["semantic_similarity"])),
            "mean_overall_score": float(np.mean(scores["overall"])),
            "std_overall_score": float(np.std(scores["overall"])),
            "num_questions": len(scores["overall"])
        }
    
    # Rank models
    ranked_models = sorted(
        aggregate_stats.items(),
        key=lambda x: x[1]["mean_overall_score"],
        reverse=True
    )
    
    return {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "judge_model": "gpt-4o-mini",
            "baseline_model": "gemini-2.0-flash-lite",
            "embedding_model": embedding_model_name,
            "num_questions": len(generations)
        },
        "question_evaluations": evaluations,
        "aggregate_statistics": aggregate_stats,
        "model_ranking": [
            {"rank": i+1, "model": name, "score": stats["mean_overall_score"]}
            for i, (name, stats) in enumerate(ranked_models)
        ]
    }


# ==================================================
# MAIN
# ==================================================

def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("RAG GENERATION EVALUATION SYSTEM")
    print("="*80)
    
    # Load evaluation questions
    print("\n1. Loading evaluation questions...")
    try:
        with open(EVALUATION_QUESTIONS_PATH, 'r') as f:
            questions_data = json.load(f)
        print(f"   ✓ Loaded {len(questions_data['questions'])} questions")
    except FileNotFoundError:
        print(f"Error: File not found at {EVALUATION_QUESTIONS_PATH}")
        return

    # Get available Ollama models
    print("\n2. Checking available Ollama models...")
    available_ollama = get_available_ollama_models()
    
    if not available_ollama:
        print("   ⚠️  No Ollama models found. Will use Gemini only.")
        ollama_models = []
    else:
        print(f"   ✓ Found {len(available_ollama)} Ollama models:")
        for i, model in enumerate(available_ollama, 1):
            print(f"      {i}. {model}")
        
        # Select models
        selection = input("\n   Enter model numbers (comma-separated) or 'all': ").strip()
        
        if selection.lower() == 'all':
            ollama_models = available_ollama
        elif selection:
            indices = []
            for x in selection.split(","):
                x = x.strip()
                if x.isdigit():
                    idx = int(x) - 1
                    if 0 <= idx < len(available_ollama):
                        indices.append(idx)
            ollama_models = [available_ollama[i] for i in indices]
        else:
            ollama_models = [available_ollama[0]] if available_ollama else []
        
        print(f"   ✓ Selected {len(ollama_models)} models")
    
    # Initialize RAG system
    print("\n3. Initializing RAG system...")
    # Pass the name of the model string, not the model object
    rag_system = RAGSystem(QDRANT_URL, COLLECTION, EMBEDDING_MODEL)
    
    # Run generation pipeline
    print("\n4. Running generation pipeline...")
    generation_results = run_generation_pipeline(
        questions_data, rag_system, ollama_models
    )
    
    # Save generation results
    print(f"\n5. Saving generation results...")
    with open(GENERATION_RESULTS_PATH, 'w') as f:
        json.dump(generation_results, f, indent=2)
    print(f"   ✓ Saved to: {GENERATION_RESULTS_PATH}")
    
    # Run evaluation pipeline
    print("\n6. Running evaluation pipeline (LLM-as-judge)...")
    # Pass the embedding model name string
    evaluation_results = run_evaluation_pipeline(generation_results, EMBEDDING_MODEL)
    
    # Save evaluation results
    print(f"\n7. Saving evaluation results...")
    with open(EVALUATION_RESULTS_PATH, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"   ✓ Saved to: {EVALUATION_RESULTS_PATH}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print("\n📊 MODEL RANKING (by overall score):")
    for rank_info in evaluation_results["model_ranking"]:
        print(f"  {rank_info['rank']}. {rank_info['model']:30s} "
              f"Score: {rank_info['score']:.2f}/10")
    
    print("\n📈 DETAILED SCORES:")
    for model, stats in evaluation_results["aggregate_statistics"].items():
        print(f"\n  {model}:")
        print(f"    Faithfulness:  {stats['mean_faithfulness']:.2f}/10")
        print(f"    Completeness:  {stats['mean_completeness']:.2f}/10")
        print(f"    Correctness:   {stats['mean_correctness']:.2f}/10")
        print(f"    Sem. Sim.:     {stats['mean_semantic_similarity_to_gemini']:.3f}")
        print(f"    Overall:       {stats['mean_overall_score']:.2f}/10 (+/- {stats['std_overall_score']:.2f})")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nGeneration results: {GENERATION_RESULTS_PATH}")
    print(f"Evaluation results: {EVALUATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()