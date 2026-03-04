"""
Automatic Question Generation for RAG Evaluation (IMPROVED)
Extracts content from PDFs and generates diverse, high-quality questions
with multi-hop reasoning, cross-page logic, difficulty verification, and quality validation.

New in this version:
- DifficultyScorer: objective computational difficulty scoring (0-100)
- DifficultyVerifier: LLM-as-Judge for independent difficulty assessment
- Difficulty-specific generation prompts (easy / medium / hard)
- Difficulty distribution enforcement with targeted regeneration
- Enhanced quality verification (LLM grounding, evidence substring check, trivial-easy rejection)
- Semantic deduplication enabled by default
- argparse for all configurable paths/settings
- Comprehensive end-of-run statistics
"""

import argparse
import os
import json
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
import pdfplumber
from tqdm import tqdm
import time
import hashlib
import numpy as np

# ==================================================
# OLLAMA LOCAL LLM CONFIGURATION
# ==================================================
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:72b")  # Recommended: qwen2.5:32b or mixtral:8x7b

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 12000  # characters per chunk with overlap
CHUNK_OVERLAP = 2000  # overlap to maintain context

# Semantic Duplicate Detection (enabled by default; requires Ollama with BGE-M3)
ENABLE_SEMANTIC_DEDUP = os.environ.get("ENABLE_SEMANTIC_DEDUP", "true").lower() == "true"
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD", "0.85"))

# LLM-as-Judge and difficulty enforcement flags
ENABLE_LLM_JUDGE = os.environ.get("ENABLE_LLM_JUDGE", "true").lower() == "true"
DIFFICULTY_ENFORCEMENT = os.environ.get("DIFFICULTY_ENFORCEMENT", "true").lower() == "true"

# Target difficulty distribution: 10% easy, 50% medium, 40% hard
TARGET_DIFFICULTY_DIST = {"easy": 0.10, "medium": 0.50, "hard": 0.40}
MAX_REGEN_ATTEMPTS = 3  # maximum regeneration attempts per difficulty level


# ==================================================
# DIFFICULTY SCORING & VERIFICATION
# ==================================================

class DifficultyScorer:
    """
    Objectively score question difficulty based on measurable linguistic and structural signals.

    Scoring scale 0-100:
      0-30  → easy
      31-65 → medium
      66-100 → hard
    """

    # Words that increase perceived difficulty
    CONDITIONAL_WORDS = {"if", "given", "assuming", "based on", "provided that", "suppose", "considering"}
    COMPARISON_WORDS  = {"compare", "contrast", "difference between", "whereas", "distinguish", "versus"}
    CAUSAL_WORDS      = {"why", "how does", "what causes", "consequence", "effect of", "result of", "leads to"}
    TEMPORAL_WORDS    = {"before", "after", "sequence", "timeline", "first then", "order of", "chronological"}
    # Words/phrases that suggest a simple single-fact lookup → decrease difficulty
    SIMPLE_LOOKUP_PATTERNS = re.compile(
        r"^\s*(what is|what are|who is|who was|where is|define|name the|list the|when did|when was)",
        re.IGNORECASE
    )

    @classmethod
    def score(cls, question: str, answer: str, context: str) -> Tuple[int, str]:
        """
        Compute a numeric difficulty score and categorical label.

        Returns:
            (score_0_to_100, difficulty_label)  where label is "easy"|"medium"|"hard"
        """
        q_lower = question.lower()
        a_lower = answer.lower()
        score = 0

        # --- Reasoning-hop estimate (0-30 pts) ---
        # Count sentences in the answer as a proxy for reasoning depth
        answer_sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
        hop_count = len(answer_sentences)
        if hop_count == 1:
            score += 5   # single fact
        elif hop_count == 2:
            score += 18  # two-step reasoning
        else:
            score += 30  # 3+ reasoning steps

        # --- Entity density in question (0-20 pts) ---
        # Naïve heuristic: count words that start with a capital letter (excl. first word)
        q_words = question.split()
        entity_count = sum(1 for w in q_words[1:] if w and w[0].isupper())
        # Also count numbers and dates
        entity_count += len(re.findall(r"\b\d+(?:[.,]\d+)*\b", question))
        score += min(20, entity_count * 4)

        # --- Complexity indicator bonuses ---
        if any(w in q_lower for w in cls.CONDITIONAL_WORDS):
            score += 10
        if any(w in q_lower for w in cls.COMPARISON_WORDS):
            score += 8
        if any(w in q_lower for w in cls.CAUSAL_WORDS):
            score += 8
        if any(w in q_lower for w in cls.TEMPORAL_WORDS):
            score += 7

        # --- Simple lookup penalty (−15 pts) ---
        if cls.SIMPLE_LOOKUP_PATTERNS.match(question):
            score = max(0, score - 15)

        # --- Answer complexity (0-15 pts) ---
        # Multi-sentence answers with numbers or cross-section cues
        has_numbers = bool(re.search(r"\d", answer))
        if has_numbers:
            score += 5
        if hop_count >= 3:
            score += 10

        # --- Cross-reference requirement (0-10 pts) ---
        # Proxy: does the context contain multiple [Page N] markers?
        page_markers = re.findall(r"\[Page \d+\]", context)
        unique_pages = len(set(page_markers))
        if unique_pages >= 3:
            score += 10
        elif unique_pages == 2:
            score += 5

        # Clamp to [0, 100]
        score = max(0, min(100, score))

        # Map to label
        if score <= 30:
            label = "easy"
        elif score <= 65:
            label = "medium"
        else:
            label = "hard"

        return score, label


class DifficultyVerifier:
    """
    LLM-as-Judge: independently verify question difficulty using a separate,
    low-temperature LLM call.  Uses the same Ollama model as the generator.
    """

    JUDGE_PROMPT_TEMPLATE = """You are a strict difficulty judge for question-answering tasks.
Given the question, its context, and the difficulty label assigned by the question generator,
determine the TRUE difficulty.

DIFFICULTY CRITERIA:
- EASY: Can be answered by finding a single fact in one sentence. Simple lookup.
  Examples: "What is X?", "When did Y happen?", "Name the Z"
- MEDIUM: Requires understanding 2 pieces of information, or understanding a concept and
  applying it. Requires reading a paragraph, not just a sentence.
- HARD: Requires synthesising 3+ pieces of information from different parts of the text,
  performing calculations, making inferences NOT directly stated, or applying complex
  reasoning chains.

Question: {question}
Context (excerpt): {context}
LLM-assigned difficulty: {original_difficulty}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "true_difficulty": "easy|medium|hard",
  "confidence": 0.0,
  "reasoning": "one sentence explanation",
  "is_mislabeled": false
}}"""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL,
                 enabled: bool = ENABLE_LLM_JUDGE):
        self.model = model
        self.base_url = base_url
        self.enabled = enabled

    def verify(self, question: str, answer: str, context: str,
               original_difficulty: str) -> Dict[str, Any]:
        """
        Ask the LLM to independently assess difficulty.

        Returns a dict with keys:
          true_difficulty, confidence, reasoning, is_mislabeled
        On any error falls back to the original label with confidence 0.
        """
        default = {
            "true_difficulty": original_difficulty,
            "confidence": 0.0,
            "reasoning": "LLM judge unavailable or disabled",
            "is_mislabeled": False,
        }
        if not self.enabled:
            return default

        # Truncate context to keep prompt manageable
        ctx_excerpt = context[:1500] if len(context) > 1500 else context

        prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            context=ctx_excerpt,
            original_difficulty=original_difficulty,
        )

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300},
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()

            result = json.loads(raw)
            # Normalise
            result["true_difficulty"] = result.get("true_difficulty", original_difficulty).lower()
            result["is_mislabeled"] = (
                result.get("is_mislabeled", False)
                or result["true_difficulty"] != original_difficulty.lower()
            )
            return result
        except Exception as exc:
            print(f"    ⚠️  LLM judge error: {exc}")
            return default


# ==================================================
# PDF EXTRACTION
# ==================================================

class PDFContentExtractor:
    """Extract comprehensive content from PDFs including text, tables, and structure"""
    
    @staticmethod
    def intelligent_chunking(text: str, max_chunk_size: int = CHUNK_SIZE, 
                           overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        """
        Intelligently chunk text to avoid cutting mid-sentence
        Returns list of chunks with metadata
        """
        chunks = []
        
        # Split by pages first to maintain page boundaries
        pages = text.split('[Page ')
        current_chunk = ""
        current_pages = []
        
        for page in pages:
            if not page.strip():
                continue
            
            # Extract page number
            try:
                page_num = int(page.split(']')[0])
                page_content = ']'.join(page.split(']')[1:])
            except (ValueError, IndexError):
                page_content = page
                page_num = len(current_pages) + 1 if current_pages else 1
            
            # Check if adding this page exceeds chunk size
            if len(current_chunk) + len(page_content) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'pages': current_pages.copy(),
                    'char_count': len(current_chunk)
                })
                
                # Start new chunk with overlap from previous
                sentences = current_chunk.split('.')
                overlap_text = '.'.join(sentences[-5:]) if len(sentences) > 5 else current_chunk[-overlap:]
                current_chunk = overlap_text + f"\n[Page {page_num}]{page_content}"
                current_pages = [page_num]
            else:
                current_chunk += f"\n[Page {page_num}]{page_content}"
                if page_num not in current_pages:
                    current_pages.append(page_num)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'pages': current_pages.copy(),
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> Dict[str, Any]:
        """Extract text and tables using pdfplumber (best for complex PDFs)"""
        content = {
            "pages": [],
            "full_text": "",
            "tables": [],
            "metadata": {},
            "chunks": []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                content["metadata"] = {
                    "num_pages": len(pdf.pages),
                    "filename": os.path.basename(pdf_path)
                }
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = {
                        "page_number": page_num,
                        "text": "",
                        "tables": []
                    }
                    
                    # Extract text
                    text = page.extract_text()
                    if text:
                        page_data["text"] = text.strip()
                        content["full_text"] += f"\n[Page {page_num}]\n{text}\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table:
                                table_data = {
                                    "page": page_num,
                                    "table_id": f"page{page_num}_table{table_idx+1}",
                                    "data": table
                                }
                                page_data["tables"].append(table_data)
                                content["tables"].append(table_data)
                                
                                # Add table to full text in readable format
                                content["full_text"] += f"\n[Table on Page {page_num}]\n"
                                for row in table:
                                    content["full_text"] += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                    
                    content["pages"].append(page_data)
                
                # Create intelligent chunks
                content["chunks"] = PDFContentExtractor.intelligent_chunking(content["full_text"])
                
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            return None
        
        return content
    
    @staticmethod
    def extract_with_pypdf2(pdf_path: str) -> Optional[str]:
        """Fallback extraction using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num}]\n{page_text}\n"
            return text
        except Exception as e:
            print(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return None


class SemanticDuplicateDetector:
    """Detect semantically similar questions using BGE-M3 embeddings"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, threshold: float = 0.85):
        self.base_url = base_url
        self.threshold = threshold
        self.question_embeddings = []  # List of (question_text, embedding)
        self.enabled = ENABLE_SEMANTIC_DEDUP
        
        if self.enabled:
            try:
                self._verify_ollama()
                print(f"✅ Semantic deduplication enabled (threshold: {threshold})")
            except Exception as e:
                print(f"⚠️  Semantic deduplication disabled: {e}")
                self.enabled = False
    
    def _verify_ollama(self):
        """Verify Ollama is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if "bge-m3:latest" not in models:
                raise RuntimeError("bge-m3:latest model not found")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Ollama: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        if not self.enabled:
            return np.array([])
        
        try:
            payload = {"model": "bge-m3:latest", "prompt": text}
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"    ⚠️  Embedding error: {e}")
            return np.array([])
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def is_duplicate(self, question: str) -> Tuple[bool, float]:
        """Check if question is semantically similar"""
        if not self.enabled or not self.question_embeddings:
            return False, 0.0
        
        new_embedding = self.get_embedding(question)
        if len(new_embedding) == 0:
            return False, 0.0
        
        max_similarity = 0.0
        for _, existing_embedding in self.question_embeddings:
            similarity = self.cosine_similarity(new_embedding, existing_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
        
        is_dup = max_similarity >= self.threshold
        return is_dup, max_similarity
    
    def add_question(self, question: str):
        """Add question to database"""
        if not self.enabled:
            return
        embedding = self.get_embedding(question)
        if len(embedding) > 0:
            self.question_embeddings.append((question, embedding))


class QuestionGenerator:
    """Generate diverse evaluation questions using Ollama local LLM"""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.generated_hashes = set()  # Track question hashes to avoid duplicates
        
        # Verify Ollama is accessible
        self._verify_ollama()
        
        # Initialize semantic duplicate detector
        self.semantic_detector = SemanticDuplicateDetector(
            base_url=OLLAMA_BASE_URL,
            threshold=SEMANTIC_SIMILARITY_THRESHOLD
        )
    
    def _verify_ollama(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            available_models = [m["name"] for m in response.json().get("models", [])]
            
            if self.model not in available_models:
                print(f"\n⚠️  WARNING: {self.model} not found in Ollama")
                print(f"Available models:")
                for m in available_models[:10]:
                    print(f"  - {m}")
                raise RuntimeError(f"Model {self.model} not available. Pull it with: ollama pull {self.model}")
            
            print(f"✅ Ollama model verified: {self.model}")
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Cannot connect to Ollama at {self.base_url}")
            print(f"Make sure Ollama is running: docker ps | grep ollama")
            raise RuntimeError(f"Ollama not accessible: {e}")
    
    def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000, 
                 retry_count: int = 0) -> str:
        """Call Ollama local LLM with retry logic"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Increased timeout for large models
            )
            
            if response.status_code == 404:
                raise Exception(f"Model {self.model} not found. Pull it with: ollama pull {self.model}")
            
            if response.status_code != 200:
                if retry_count < MAX_RETRIES:
                    wait_time = RETRY_DELAY * (2 ** retry_count)
                    print(f"  ⚠️ API Error {response.status_code}. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    return self.call_llm(prompt, temperature, max_tokens, retry_count + 1)
                else:
                    raise Exception(f"API Error {response.status_code}: {response.text[:200]}")
            
            result = response.json()
            return result.get("response", "")
            
        except requests.Timeout:
            if retry_count < MAX_RETRIES:
                print(f"  ⚠️ Request timeout. Retrying...")
                time.sleep(RETRY_DELAY)
                return self.call_llm(prompt, temperature, max_tokens, retry_count + 1)
            else:
                raise Exception("Request timeout after max retries")
        except Exception as e:
            if retry_count < MAX_RETRIES and "Model" not in str(e):
                print(f"  ⚠️ Error: {str(e)[:100]}. Retrying...")
                time.sleep(RETRY_DELAY)
                return self.call_llm(prompt, temperature, max_tokens, retry_count + 1)
            else:
                raise Exception(f"Failed to generate response after {MAX_RETRIES} retries: {str(e)}")
    
    def get_question_hash(self, question: str) -> str:
        """Generate hash for duplicate detection"""
        # Normalize question (lowercase, remove punctuation)
        normalized = question.lower().strip().replace('?', '').replace('.', '')
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, question: str) -> bool:
        """Check if question is an exact duplicate (hash-based)"""
        q_hash = self.get_question_hash(question)
        
        # Exact duplicate check
        if q_hash in self.generated_hashes:
            return True
        
        # Add to seen hashes
        self.generated_hashes.add(q_hash)
        return False
    
    def parse_json_response(self, response: str) -> List[Dict]:
        """Robustly parse JSON from LLM response"""
        # Clean markdown formatting
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        # Try to find JSON array in response
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx+1]
        
        try:
            questions = json.loads(response)
            if not isinstance(questions, list):
                raise ValueError("Response is not a list")
            return questions
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON parsing error: {e}")
            print(f"  Response preview: {response[:200]}")
            return []
    
    def verify_question_quality(self, question: Dict, context: str,
                                 scorer: "DifficultyScorer" = None,
                                 verifier: "DifficultyVerifier" = None,
                                 rejection_stats: Dict = None) -> Tuple[bool, str]:
        """
        Verify question quality and answerability.

        Enhanced checks:
        - Required-field and length validation
        - Exact and semantic duplicate detection
        - Bag-of-words answer grounding (≥30% overlap)
        - Evidence snippet substring verification
        - Trivial-easy-labeled-hard rejection
        - Computational difficulty consistency (within 1 level)
        - LLM-as-Judge difficulty re-labeling

        Returns (is_valid, reason)
        """
        if rejection_stats is None:
            rejection_stats = {}

        def _reject(reason: str) -> Tuple[bool, str]:
            rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
            return False, reason

        # --- Required fields ---
        required_fields = ['question', 'answer', 'question_type', 'difficulty']
        for field in required_fields:
            if field not in question or not question[field]:
                return _reject(f"Missing required field: {field}")

        q_text = question['question'].strip()
        a_text = question['answer'].strip()
        difficulty = question['difficulty'].lower()

        # --- Length checks ---
        if len(q_text) < 10:
            return _reject("Question too short")
        if len(q_text) > 300:
            return _reject("Question too long")
        if len(a_text) < 5:
            return _reject("Answer too short")

        # --- Trivial-easy labeled as hard ---
        trivial_prefixes = re.compile(
            r"^\s*(what is|what are|define|name the|list the|who is|who was)",
            re.IGNORECASE,
        )
        if difficulty == "hard" and trivial_prefixes.match(q_text):
            return _reject("Trivially easy question labeled as hard")

        # --- Exact duplicate ---
        if self.is_duplicate(q_text):
            return _reject("Exact duplicate question")

        # --- Semantic duplicate ---
        if self.semantic_detector.enabled:
            is_semantic_dup, similarity = self.semantic_detector.is_duplicate(q_text)
            if is_semantic_dup:
                return _reject(f"Semantic duplicate (similarity: {similarity:.3f})")

        # --- Bag-of-words grounding ---
        answer_words = set(a_text.lower().split())
        context_words = set(context.lower().split())
        if len(answer_words) > 5:
            overlap = len(answer_words.intersection(context_words))
            if overlap / len(answer_words) < 0.3:
                return _reject("Answer not grounded in context")

        # --- Evidence snippet substring verification ---
        snippets = question.get("evidence_snippets", [])
        if snippets:
            ctx_lower = context.lower()
            verified_snippets = []
            for snip in snippets:
                snip_str = str(snip).strip()
                # Accept if at least a 15-char substring is present
                check = snip_str[:15].lower() if len(snip_str) >= 15 else snip_str.lower()
                if check and check in ctx_lower:
                    verified_snippets.append(snip_str)
            if not verified_snippets and len(snippets) > 0:
                return _reject("Evidence snippets not found in source context")
            question["evidence_snippets"] = verified_snippets

        # --- Computational difficulty scoring ---
        comp_score, comp_label = DifficultyScorer.score(q_text, a_text, context)
        levels = ["easy", "medium", "hard"]
        diff_idx = levels.index(difficulty) if difficulty in levels else 1
        comp_idx = levels.index(comp_label)
        # Reject if more than 1 level apart
        if abs(diff_idx - comp_idx) > 1:
            return _reject(
                f"Difficulty label '{difficulty}' inconsistent with computational score "
                f"{comp_score} (suggests '{comp_label}')"
            )

        # --- LLM-as-Judge ---
        original_difficulty = difficulty
        llm_judge_result = {"true_difficulty": difficulty, "confidence": 0.0,
                            "reasoning": "judge disabled", "is_mislabeled": False}
        if verifier is not None:
            llm_judge_result = verifier.verify(q_text, a_text, context, difficulty)
            if llm_judge_result.get("is_mislabeled"):
                judge_diff = llm_judge_result["true_difficulty"]
                # If both comp AND judge agree → relabel (don't reject)
                if judge_diff == comp_label:
                    print(f"    🔄 Relabeling '{difficulty}' → '{judge_diff}' "
                          f"(judge+comp agree, confidence {llm_judge_result['confidence']:.2f})")
                    question["difficulty"] = judge_diff
                    rejection_stats.setdefault("_relabeled", 0)
                    rejection_stats["_relabeled"] += 1
                elif llm_judge_result.get("confidence", 0) >= 0.85:
                    # High-confidence judge overrides regardless
                    print(f"    🔄 Relabeling '{difficulty}' → '{judge_diff}' "
                          f"(high-confidence judge: {llm_judge_result['confidence']:.2f})")
                    question["difficulty"] = judge_diff
                    rejection_stats.setdefault("_relabeled", 0)
                    rejection_stats["_relabeled"] += 1

        # --- Attach difficulty scores to question ---
        question["difficulty_scores"] = {
            "computational_score": comp_score,
            "llm_judge_difficulty": llm_judge_result["true_difficulty"],
            "llm_judge_confidence": llm_judge_result.get("confidence", 0.0),
            "original_llm_difficulty": original_difficulty,
            "was_relabeled": question["difficulty"] != original_difficulty,
            "relabel_reason": (
                f"Computational score {comp_score} and LLM judge both indicate "
                f"'{llm_judge_result['true_difficulty']}'"
                if question["difficulty"] != original_difficulty else ""
            ),
        }
        question["quality_checks"] = {
            "answer_grounded": True,
            "evidence_verified": bool(snippets),
            "difficulty_verified": True,
            "grounding_score": round(
                len(answer_words.intersection(context_words)) / max(len(answer_words), 1), 2
            ) if answer_words else 0.0,
        }

        # All checks passed - register in dedup stores
        self.semantic_detector.add_question(q_text)
        return True, "Valid"

    def _select_diverse_questions(self, questions: List[Dict], target_count: int) -> List[Dict]:
        """Select a diverse subset of questions, proportionally by type."""
        by_type: Dict[str, List[Dict]] = {}
        for q in questions:
            q_type = q.get("question_type", "other")
            by_type.setdefault(q_type, []).append(q)

        questions_per_type = max(1, target_count // len(by_type))
        selected: List[Dict] = []
        for q_list in by_type.values():
            selected.extend(q_list[:questions_per_type])

        remaining = target_count - len(selected)
        if remaining > 0:
            in_selected = set(id(q) for q in selected)
            extras = [q for q in questions if id(q) not in in_selected]
            selected.extend(extras[:remaining])

        return selected[:target_count]

    # ------------------------------------------------------------------
    # Difficulty-specific generation prompts
    # ------------------------------------------------------------------

    def _create_easy_prompt(self, doc_text: str, num_questions: int, pages: List[int]) -> str:
        """Prompt for EASY single-fact lookup questions."""
        page_info = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
        return f"""You are creating EASY evaluation questions for a RAG benchmark.

DOCUMENT CONTENT ({page_info}):
{doc_text}

RULES FOR EASY QUESTIONS:
- The answer MUST be found in exactly ONE sentence of the document.
- Prefer "What is", "Who is", "When did", "Define", "Name the" style questions.
- No reasoning, inference, or multi-step lookup required.
- Do NOT include questions that require reading more than one sentence.

Generate EXACTLY {num_questions} EASY questions.
Return ONLY a valid JSON array:

[
  {{
    "question": "Single-fact question text",
    "answer": "Direct answer from one sentence",
    "question_type": "single-hop|definition|list",
    "difficulty": "easy",
    "reasoning_steps": "",
    "page_reference": "Page X",
    "evidence_snippets": ["Exact sentence from document"]
  }}
]

Generate EXACTLY {num_questions} easy questions now:"""

    def _create_medium_prompt(self, doc_text: str, num_questions: int, pages: List[int]) -> str:
        """Prompt for MEDIUM questions requiring 2 pieces of information."""
        page_info = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
        return f"""You are creating MEDIUM difficulty evaluation questions for a RAG benchmark.

DOCUMENT CONTENT ({page_info}):
{doc_text}

RULES FOR MEDIUM QUESTIONS:
- The answer REQUIRES understanding EXACTLY 2 pieces of information from the document.
- The answerer must read and understand at least one FULL PARAGRAPH (not just one sentence).
- Suitable types: numerical/statistical, definition+application, list+explanation, causal.
- Avoid pure single-sentence lookups (too easy) and 3+-hop reasoning (too hard).

Generate EXACTLY {num_questions} MEDIUM questions.
Return ONLY a valid JSON array:

[
  {{
    "question": "Question requiring 2 pieces of information",
    "answer": "Answer synthesising 2 pieces of evidence (2-3 sentences)",
    "question_type": "numerical|definition|list|causal|comparison",
    "difficulty": "medium",
    "reasoning_steps": "Step 1: find X. Step 2: apply/combine with Y.",
    "page_reference": "Page X or Pages X-Y",
    "evidence_snippets": ["Quote 1", "Quote 2"]
  }}
]

Generate EXACTLY {num_questions} medium questions now:"""

    def _create_hard_prompt(self, doc_text: str, num_questions: int, pages: List[int]) -> str:
        """Prompt for HARD questions requiring 3+ facts, inference, or cross-section reasoning."""
        page_info = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
        return f"""You are creating HARD evaluation questions for a RAG benchmark.

DOCUMENT CONTENT ({page_info}):
{doc_text}

RULES FOR HARD QUESTIONS:
- The answer CANNOT be found in any single paragraph.
- It requires connecting information from AT LEAST 2 different sections or pages.
- OR it requires performing a calculation, making an inference NOT directly stated, or
  following a complex causal/temporal reasoning chain of 3+ steps.
- Suitable types: multi-hop reasoning, cross-page logic, numerical calculation, causal chain,
  comparison across sections.
- Do NOT ask questions answerable from a single sentence or paragraph.

Generate EXACTLY {num_questions} HARD questions.
Return ONLY a valid JSON array:

[
  {{
    "question": "Complex question requiring multi-section synthesis or inference",
    "answer": "Detailed answer connecting 3+ facts (3-5 sentences)",
    "question_type": "multi-hop|cross-page|numerical|causal|comparison|temporal",
    "difficulty": "hard",
    "reasoning_steps": "Step 1: ... Step 2: ... Step 3: ... connect to conclude.",
    "page_reference": "Pages X-Y",
    "evidence_snippets": ["Quote from section A", "Quote from section B", "Quote from section C"]
  }}
]

Generate EXACTLY {num_questions} hard questions now:"""

    def _create_advanced_prompt(self, doc_text: str, num_questions: int, pages: List[int]) -> str:
        """Mixed-difficulty prompt (used as fallback/for diversity)."""
        page_info = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"

        prompt = f"""You are an expert at creating ADVANCED evaluation questions for Retrieval-Augmented Generation (RAG) systems.

Given the following document content, generate EXACTLY {num_questions} HIGH-QUALITY, DIVERSE questions.

DOCUMENT CONTENT ({page_info}):
{doc_text}

CRITICAL REQUIREMENTS:

1. QUESTION TYPES (Must include mix of):
   - **Single-hop factoid**: Direct fact from one location (who, what, when, where)
   - **Multi-hop reasoning**: Requires connecting 2-3 pieces of information from different sections
   - **Cross-page logic**: Answer requires information from multiple pages (if document spans pages)
   - **Numerical/statistical**: Numbers, percentages, calculations, comparisons
   - **Definition/conceptual**: Explain concepts, define terms
   - **List/enumeration**: Multiple items, steps, or components
   - **Causal reasoning**: Why/how questions requiring logical inference
   - **Comparison**: Compare/contrast multiple concepts or entities
   - **Temporal reasoning**: Sequence, timeline, before/after relationships

2. QUESTION QUALITY STANDARDS:
   - Must be answerable ONLY from provided document (no external knowledge)
   - Must have ONE clear, verifiable answer
   - Answer must cite specific evidence from document
   - Questions should vary in difficulty (easy: 10%, medium: 50%, hard: 40%)
   - Avoid yes/no questions unless they require complex reasoning
   - Avoid ambiguous or trick questions

3. MULTI-HOP EXAMPLES:
   ❌ Simple: "What is X?"
   ✅ Multi-hop: "Based on the process described in section A and the constraints in section B, what would be the outcome if X occurs?"

   ❌ Simple: "When did Y happen?"
   ✅ Multi-hop: "Given that Y happened in [year] and Z takes [duration], when would Z be completed?"

4. OUTPUT FORMAT (JSON):
Return ONLY a valid JSON array with EXACTLY {num_questions} questions:

[
  {{
    "question": "Clear, specific question text here",
    "answer": "Concise answer with key evidence (2-4 sentences max)",
    "question_type": "single-hop|multi-hop|cross-page|numerical|definition|list|causal|comparison|temporal",
    "difficulty": "easy|medium|hard",
    "reasoning_steps": "For multi-hop: briefly explain the reasoning chain (1 sentence)",
    "page_reference": "Page X or Pages X-Y",
    "evidence_snippets": ["Key quote 1", "Key quote 2"]
  }},
  ...
]

5. VALIDATION CHECKLIST:
   ✓ Each question is unique and non-redundant
   ✓ Answer is directly supported by document text
   ✓ Mix of question types and difficulties
   ✓ At least 30% are multi-hop or reasoning questions
   ✓ Evidence snippets are actual quotes from document

Generate EXACTLY {num_questions} questions now following ALL requirements:"""

        return prompt

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------

    def _generate_for_difficulty(self, chunk_text: str, chunk_pages: List[int],
                                  difficulty: str, num_questions: int,
                                  scorer: "DifficultyScorer",
                                  verifier: "DifficultyVerifier",
                                  rejection_stats: Dict) -> List[Dict]:
        """
        Generate and validate questions for a single difficulty level.
        Returns list of validated question dicts.
        """
        prompt_builders = {
            "easy":   self._create_easy_prompt,
            "medium": self._create_medium_prompt,
            "hard":   self._create_hard_prompt,
        }
        build_prompt = prompt_builders.get(difficulty, self._create_advanced_prompt)
        prompt = build_prompt(chunk_text, num_questions, chunk_pages)

        try:
            response = self.call_llm(prompt, temperature=0.7, max_tokens=2500)
            questions = self.parse_json_response(response)
        except Exception as exc:
            print(f"    ❌ Error generating {difficulty} questions: {exc}")
            return []

        valid = []
        for q in questions:
            # Force the difficulty label to match the prompt we used
            q["difficulty"] = difficulty
            ok, reason = self.verify_question_quality(
                q, chunk_text, scorer=scorer, verifier=verifier,
                rejection_stats=rejection_stats
            )
            if ok:
                valid.append(q)
            else:
                print(f"    ⚠️  Rejected ({difficulty}): {reason}")
        return valid

    def generate_questions_from_document(self, content: Dict[str, Any], doc_id: str,
                                         num_questions: int = 10,
                                         scorer: "DifficultyScorer" = None,
                                         verifier: "DifficultyVerifier" = None,
                                         rejection_stats: Dict = None) -> Tuple[List[Dict], Dict]:
        """
        Generate diverse questions from document content with quality and difficulty verification.

        Returns:
            (questions_list, per_difficulty_stats_dict)
        """
        if rejection_stats is None:
            rejection_stats = {}

        # Use chunks for generation
        chunks = content.get("chunks", [])
        if not chunks and content.get("full_text"):
            chunks = [{
                'text': content["full_text"][:CHUNK_SIZE],
                'pages': list(range(1, content["metadata"].get("num_pages", 1) + 1)),
                'char_count': len(content["full_text"]),
            }]

        # Compute per-difficulty targets
        targets = {d: max(1, round(num_questions * frac))
                   for d, frac in TARGET_DIFFICULTY_DIST.items()}
        # Adjust rounding so sum == num_questions
        diff_sum = sum(targets.values())
        if diff_sum != num_questions:
            targets["medium"] += num_questions - diff_sum

        all_questions: List[Dict] = []
        diff_buckets: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}

        questions_per_chunk = max(1, num_questions // len(chunks)) if len(chunks) > 1 else num_questions

        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = chunk['text']
            chunk_pages = chunk['pages']
            print(f"    📄 Processing chunk {chunk_idx + 1}/{len(chunks)} (Pages: {chunk_pages})")

            # Generate per difficulty
            per_diff = max(1, questions_per_chunk // 3)
            for difficulty in ("easy", "medium", "hard"):
                n = max(1, round(questions_per_chunk * TARGET_DIFFICULTY_DIST[difficulty]))
                qs = self._generate_for_difficulty(
                    chunk_text, chunk_pages, difficulty, n,
                    scorer, verifier, rejection_stats
                )
                diff_buckets[difficulty].extend(qs)

        # --- Difficulty distribution enforcement ---
        if DIFFICULTY_ENFORCEMENT:
            for difficulty, target in targets.items():
                current = len(diff_buckets[difficulty])
                attempt = 0
                while current < target and attempt < MAX_REGEN_ATTEMPTS:
                    attempt += 1
                    shortfall = target - current
                    print(f"    🔁 Regen attempt {attempt}/{MAX_REGEN_ATTEMPTS} for "
                          f"'{difficulty}' (need {shortfall} more)")
                    # Use a random chunk for regeneration
                    chunk = chunks[attempt % len(chunks)]
                    extra = self._generate_for_difficulty(
                        chunk['text'], chunk['pages'], difficulty, shortfall,
                        scorer, verifier, rejection_stats
                    )
                    diff_buckets[difficulty].extend(extra)
                    current = len(diff_buckets[difficulty])

                if len(diff_buckets[difficulty]) < target:
                    print(f"    ⚠️  Could not meet target for '{difficulty}': "
                          f"have {len(diff_buckets[difficulty])}, need {target}")

        # Assemble final list (respect per-difficulty targets)
        for difficulty, target in targets.items():
            bucket = diff_buckets[difficulty]
            all_questions.extend(bucket[:target])
            leftover = bucket[target:]
            # Any overflow added at the end if we're under total
            if len(all_questions) < num_questions:
                all_questions.extend(leftover[:num_questions - len(all_questions)])

        # Trim or keep to num_questions
        if len(all_questions) > num_questions:
            all_questions = self._select_diverse_questions(all_questions, num_questions)

        # Attach metadata
        for idx, q in enumerate(all_questions, start=1):
            q["id"] = f"{doc_id}_q{idx:03d}"
            q["source_document"] = content["metadata"]["filename"]
            if "page_reference" not in q or not q["page_reference"]:
                q["page_reference"] = "Multiple pages" if len(chunks) > 1 else "Page 1"

        diff_stats = {d: len([q for q in all_questions if q.get("difficulty") == d])
                      for d in ("easy", "medium", "hard")}
        return all_questions, diff_stats


def process_dataset(dataset_path: str, output_path: str, questions_per_doc: int = 10):
    """Process all PDFs in dataset and generate evaluation questions."""

    print("=" * 80)
    print("RAG EVALUATION QUESTION GENERATOR - OLLAMA VERSION")
    print("=" * 80)
    print(f"Dataset Path:          {dataset_path}")
    print(f"Questions per Document:{questions_per_doc}")
    print(f"Output:                {output_path}")
    print(f"LLM Model:             {OLLAMA_MODEL}")
    print(f"Max Retries:           {MAX_RETRIES}")
    print(f"Chunk Size:            {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
    print(f"Semantic dedup:        {ENABLE_SEMANTIC_DEDUP}")
    print(f"LLM Judge:             {ENABLE_LLM_JUDGE}")
    print(f"Difficulty enforcement:{DIFFICULTY_ENFORCEMENT}")
    print("=" * 80)

    # Initialize components
    extractor = PDFContentExtractor()
    generator = QuestionGenerator(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    scorer = DifficultyScorer()
    verifier = DifficultyVerifier(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                                  enabled=ENABLE_LLM_JUDGE)

    # Find all PDF files
    pdf_files = list(Path(dataset_path).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {dataset_path}")
        return

    print(f"📚 Found {len(pdf_files)} PDF files\n")

    # Process each PDF
    all_questions: List[Dict] = []
    doc_count = 0
    failed_docs = []
    global_rejection_stats: Dict = {}

    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
        doc_id = f"doc{idx + 1:03d}"
        print(f"\n📖 Processing: {pdf_path.name}")

        try:
            print("  📄 Extracting content (text, tables)...")
            content = extractor.extract_with_pdfplumber(str(pdf_path))

            if not content or not content["full_text"].strip():
                print("  ⚠️  Failed to extract content, trying PyPDF2...")
                text = extractor.extract_with_pypdf2(str(pdf_path))
                if not text:
                    print(f"  ❌ Skipping {pdf_path.name} - no content extracted")
                    failed_docs.append((pdf_path.name, "No content extracted"))
                    continue
                content = {
                    "full_text": text,
                    "metadata": {"filename": pdf_path.name, "num_pages": "unknown"},
                    "pages": [],
                    "tables": [],
                    "chunks": extractor.intelligent_chunking(text),
                }

            print(f"  ✅ Extracted {len(content['full_text'])} characters from "
                  f"{content['metadata']['num_pages']} pages")
            print(f"  📊 Created {len(content.get('chunks', []))} intelligent chunks")
            if content.get("tables"):
                print(f"  📋 Found {len(content['tables'])} tables")

            print(f"  🤖 Generating {questions_per_doc} questions using {OLLAMA_MODEL}...")
            doc_rejection: Dict = {}
            questions, diff_stats = generator.generate_questions_from_document(
                content, doc_id, num_questions=questions_per_doc,
                scorer=scorer, verifier=verifier, rejection_stats=doc_rejection
            )

            # Merge per-doc rejection stats into global
            for k, v in doc_rejection.items():
                global_rejection_stats[k] = global_rejection_stats.get(k, 0) + v

            if questions:
                all_questions.extend(questions)
                doc_count += 1
                print(f"  ✅ Generated {len(questions)} validated questions "
                      f"| easy:{diff_stats.get('easy', 0)} "
                      f"medium:{diff_stats.get('medium', 0)} "
                      f"hard:{diff_stats.get('hard', 0)}")
            else:
                print("  ⚠️  No valid questions generated")
                failed_docs.append((pdf_path.name, "No valid questions generated"))

        except Exception as exc:
            print(f"  ❌ Error processing {pdf_path.name}: {str(exc)[:100]}")
            failed_docs.append((pdf_path.name, str(exc)[:100]))
            continue

    # Analyse question distribution
    question_stats = analyze_questions(all_questions)

    # Create final output
    output_data = {
        "dataset_info": {
            "total_documents": doc_count,
            "total_questions": len(all_questions),
            "failed_documents": len(failed_docs),
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": f"ollama-{OLLAMA_MODEL}",
            "questions_per_document": questions_per_doc,
            "dataset_path": dataset_path,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "question_statistics": question_stats,
        },
        "questions": all_questions,
        "failed_documents": [{"filename": n, "reason": r} for n, r in failed_docs],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # --- Comprehensive end-of-run statistics ---
    total_q = len(all_questions)
    relabeled = global_rejection_stats.pop("_relabeled", 0)
    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)
    print(f"Documents Processed:      {doc_count}")
    print(f"Total Questions:          {total_q}")
    print(f"Failed Documents:         {len(failed_docs)}")
    print(f"Output saved to:          {output_path}")
    print("=" * 80)

    print("\n📊 DIFFICULTY DISTRIBUTION (actual vs target):")
    print("-" * 80)
    by_diff = question_stats.get("by_difficulty", {})
    for diff_level, target_frac in TARGET_DIFFICULTY_DIST.items():
        actual = by_diff.get(diff_level, 0)
        actual_pct = round(actual / max(total_q, 1) * 100, 1)
        target_pct = round(target_frac * 100, 1)
        print(f"  {diff_level:<8}: {actual:>4} ({actual_pct:>5.1f}%)  target: {target_pct:.0f}%")
    print("-" * 80)

    print("\n📊 RELABELING & REJECTION STATS:")
    print("-" * 80)
    print(f"  Questions relabeled by LLM Judge: {relabeled} "
          f"({round(relabeled / max(total_q, 1) * 100, 1)}%)")
    for reason, count in sorted(global_rejection_stats.items(), key=lambda x: -x[1]):
        print(f"  Rejected — {reason}: {count}")
    print("-" * 80)

    print("\n📊 AVERAGE COMPUTATIONAL SCORE BY DIFFICULTY:")
    print("-" * 80)
    for diff_level in ("easy", "medium", "hard"):
        scores = [q.get("difficulty_scores", {}).get("computational_score", 0)
                  for q in all_questions if q.get("difficulty") == diff_level]
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        print(f"  {diff_level:<8}: avg computational score = {avg}")
    print("-" * 80)

    judge_total = sum(1 for q in all_questions if q.get("difficulty_scores", {}).get("llm_judge_confidence", 0) > 0)
    judge_agree = sum(
        1 for q in all_questions
        if (q.get("difficulty_scores", {}).get("llm_judge_difficulty") ==
            q.get("difficulty_scores", {}).get("original_llm_difficulty"))
        and q.get("difficulty_scores", {}).get("llm_judge_confidence", 0) > 0
    )
    if judge_total > 0:
        print(f"\n📊 LLM JUDGE AGREEMENT RATE: "
              f"{judge_agree}/{judge_total} ({round(judge_agree / judge_total * 100, 1)}%)")

    print("\n📊 QUESTION STATISTICS:")
    print("-" * 80)
    for key, value in question_stats.items():
        print(f"  {key}: {value}")
    print("-" * 80)

    # Sample questions
    if all_questions:
        print("\n🔍 SAMPLE QUESTIONS:")
        print("-" * 80)
        for q_type in ["multi-hop", "cross-page", "causal", "single-hop"]:
            matching = [q for q in all_questions if q.get("question_type", "").startswith(q_type)]
            if matching:
                q = matching[0]
                print(f"\n[{q['question_type'].upper()}] Difficulty: {q['difficulty']}")
                print(f"Q: {q['question']}")
                print(f"A: {q['answer'][:150]}...")
                if q.get("reasoning_steps"):
                    print(f"Reasoning: {q['reasoning_steps']}")
                ds = q.get("difficulty_scores", {})
                if ds:
                    print(f"Comp score: {ds.get('computational_score')}  "
                          f"Judge: {ds.get('llm_judge_difficulty')}  "
                          f"Relabeled: {ds.get('was_relabeled')}")
        print("-" * 80)

    if failed_docs:
        print("\n⚠️  FAILED DOCUMENTS:")
        print("-" * 80)
        for name, reason in failed_docs:
            print(f"  • {name}: {reason}")
        print("-" * 80)


def analyze_questions(questions: List[Dict]) -> Dict[str, Any]:
    """Analyze question distribution and statistics."""
    stats: Dict[str, Any] = {
        "by_type": {},
        "by_difficulty": {},
        "multi_hop_percentage": 0,
        "avg_answer_length": 0,
        "questions_with_evidence": 0,
    }

    total = len(questions)
    if total == 0:
        return stats

    answer_lengths = []
    multi_hop_count = 0
    evidence_count = 0

    for q in questions:
        q_type = q.get("question_type", "unknown")
        stats["by_type"][q_type] = stats["by_type"].get(q_type, 0) + 1

        difficulty = q.get("difficulty", "unknown")
        stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1

        if any(t in q_type for t in ("multi-hop", "cross-page", "causal")):
            multi_hop_count += 1

        if "answer" in q:
            answer_lengths.append(len(q["answer"]))

        if q.get("evidence_snippets"):
            evidence_count += 1

    stats["multi_hop_percentage"] = round(multi_hop_count / total * 100, 2)
    stats["avg_answer_length"] = (
        round(sum(answer_lengths) / len(answer_lengths)) if answer_lengths else 0
    )
    stats["questions_with_evidence"] = evidence_count

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RAG evaluation questions from PDF documents using Ollama."
    )
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("DATASET_PATH", "./DATASET"),
        help="Path to directory containing PDF files (default: ./DATASET)",
    )
    parser.add_argument(
        "--output-path",
        default=os.environ.get(
            "OUTPUT_PATH", "./evaluation_questions_Qwen.json"
        ),
        help="Path for output JSON file (default: ./evaluation_questions_Qwen.json)",
    )
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=int(os.environ.get("QUESTIONS_PER_DOC", "10")),
        help="Number of questions to generate per document (default: 10)",
    )
    args = parser.parse_args()

    print(f"🤖 Using Ollama Model: {OLLAMA_MODEL}")
    print(f"🌐 Ollama URL:         {OLLAMA_BASE_URL}")

    # Verify Ollama is accessible before starting
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        print(f"✅ Ollama is running at {OLLAMA_BASE_URL}\n")
    except Exception as exc:
        print(f"❌ ERROR: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print(f"   Error: {exc}")
        print("\nMake sure Ollama is running:")
        print("   docker ps | grep ollama")
        print("   OR: systemctl status ollama")
        raise SystemExit(1)

    process_dataset(args.dataset_path, args.output_path, args.questions_per_doc)