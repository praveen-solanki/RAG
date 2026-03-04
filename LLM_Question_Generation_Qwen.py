"""
Automatic Question Generation for RAG Evaluation (IMPROVED)
Extracts content from PDFs and generates diverse, high-quality questions
with multi-hop reasoning, cross-page logic, and quality verification
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
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

print(f"🤖 Using Ollama Model: {OLLAMA_MODEL}")
print(f"🌐 Ollama URL: {OLLAMA_BASE_URL}")

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 12000  # characters per chunk with overlap
CHUNK_OVERLAP = 2000  # overlap to maintain context

# Semantic Duplicate Detection (Optional - requires Ollama with BGE-M3)
ENABLE_SEMANTIC_DEDUP = os.environ.get("ENABLE_SEMANTIC_DEDUP", "false").lower() == "true"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_SIMILARITY_THRESHOLD", "0.85"))


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
    def extract_with_pypdf2(pdf_path: str) -> str:
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
    
    def is_duplicate(self, question: str, threshold: float = 0.8) -> bool:
        """Check if question is duplicate or too similar"""
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
    
    def verify_question_quality(self, question: Dict, context: str) -> Tuple[bool, str]:
        """
        Verify question quality and answerability
        Returns (is_valid, reason)
        """
        # Check required fields
        required_fields = ['question', 'answer', 'question_type', 'difficulty']
        for field in required_fields:
            if field not in question or not question[field]:
                return False, f"Missing required field: {field}"
        
        # Check question length (too short or too long)
        q_text = question['question'].strip()
        if len(q_text) < 10:
            return False, "Question too short"
        if len(q_text) > 300:
            return False, "Question too long"
        
        # Check answer length
        a_text = question['answer'].strip()
        if len(a_text) < 5:
            return False, "Answer too short"
        
        # Check for exact duplicate (hash-based)
        if self.is_duplicate(q_text):
            return False, "Exact duplicate question"
        
        # Check for semantic duplicate (embedding-based)
        if self.semantic_detector.enabled:
            is_semantic_dup, similarity = self.semantic_detector.is_duplicate(q_text)
            if is_semantic_dup:
                return False, f"Semantic duplicate (similarity: {similarity:.3f})"
        
        # Verify answer appears to be grounded in context (basic check)
        # Extract key terms from answer
        answer_words = set(a_text.lower().split())
        context_words = set(context.lower().split())
        
        # At least 30% of answer words should appear in context
        if len(answer_words) > 5:
            overlap = len(answer_words.intersection(context_words))
            if overlap / len(answer_words) < 0.3:
                return False, "Answer not grounded in context"
        
        # All checks passed - add to both hash set and semantic detector
        self.semantic_detector.add_question(q_text)
        
        return True, "Valid"
    
    def generate_questions_from_document(self, content: Dict[str, Any], doc_id: str, 
                                        num_questions: int = 8) -> List[Dict]:
        """Generate diverse questions from document content with quality verification"""
        
        all_questions = []
        
        # Use chunks for generation
        chunks = content.get("chunks", [])
        if not chunks and content.get("full_text"):
            # Fallback: create single chunk
            chunks = [{
                'text': content["full_text"][:CHUNK_SIZE],
                'pages': list(range(1, content["metadata"].get("num_pages", 1) + 1)),
                'char_count': len(content["full_text"])
            }]
        
        # Generate questions per chunk
        questions_per_chunk = max(1, num_questions // len(chunks)) if len(chunks) > 1 else num_questions
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = chunk['text']
            chunk_pages = chunk['pages']
            
            print(f"    📄 Processing chunk {chunk_idx + 1}/{len(chunks)} (Pages: {chunk_pages})")
            
            prompt = self._create_advanced_prompt(chunk_text, questions_per_chunk, chunk_pages)
            
            try:
                response = self.call_llm(prompt, temperature=0.7, max_tokens=2500)
                questions = self.parse_json_response(response)
                
                if not questions:
                    print(f"    ⚠️ No valid questions generated for chunk {chunk_idx + 1}")
                    continue
                
                # Validate and filter questions
                valid_questions = []
                for q in questions:
                    is_valid, reason = self.verify_question_quality(q, chunk_text)
                    if is_valid:
                        valid_questions.append(q)
                    else:
                        print(f"    ⚠️ Rejected question: {reason}")
                
                print(f"    ✅ Generated {len(valid_questions)} valid questions from chunk {chunk_idx + 1}")
                all_questions.extend(valid_questions)
                
            except Exception as e:
                print(f"    ❌ Error generating questions for chunk {chunk_idx + 1}: {e}")
                continue
        
        # Add metadata to each question
        for idx, q in enumerate(all_questions, start=1):
            q["id"] = f"{doc_id}_q{idx:03d}"
            q["source_document"] = content["metadata"]["filename"]
            
            # Ensure page_reference exists
            if "page_reference" not in q or not q["page_reference"]:
                q["page_reference"] = "Multiple pages" if len(chunks) > 1 else "Page 1"
        
        # Limit to requested number
        if len(all_questions) > num_questions:
            # Prioritize diversity: keep questions of different types
            all_questions = self._select_diverse_questions(all_questions, num_questions)
        
        return all_questions
    
    def _select_diverse_questions(self, questions: List[Dict], target_count: int) -> List[Dict]:
        """Select diverse subset of questions"""
        # Group by type
        by_type = {}
        for q in questions:
            q_type = q.get('question_type', 'other')
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(q)
        
        # Select proportionally from each type
        selected = []
        questions_per_type = max(1, target_count // len(by_type))
        
        for q_type, q_list in by_type.items():
            selected.extend(q_list[:questions_per_type])
        
        # Fill remaining slots
        remaining = target_count - len(selected)
        if remaining > 0:
            all_remaining = [q for q in questions if q not in selected]
            selected.extend(all_remaining[:remaining])
        
        return selected[:target_count]
    
    def _create_advanced_prompt(self, doc_text: str, num_questions: int, pages: List[int]) -> str:
        """Create advanced prompt for multi-hop and logical reasoning questions"""
        
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


def process_dataset(dataset_path: str, output_path: str, questions_per_doc: int = 8):
    """Process all PDFs in dataset and generate evaluation questions"""
    
    print("="*80)
    print("RAG EVALUATION QUESTION GENERATOR - OLLAMA VERSION")
    print("="*80)
    print(f"Dataset Path: {dataset_path}")
    print(f"Questions per Document: {questions_per_doc}")
    print(f"Output: {output_path}")
    print(f"LLM Model: {OLLAMA_MODEL}")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"Chunk Size: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
    print("="*80)
    
    # Initialize components
    extractor = PDFContentExtractor()
    generator = QuestionGenerator(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Find all PDF files
    pdf_files = list(Path(dataset_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {dataset_path}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files\n")
    
    # Process each PDF
    all_questions = []
    doc_count = 0
    failed_docs = []
    
    for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
        doc_id = f"doc{idx+1:03d}"
        
        print(f"\n📖 Processing: {pdf_path.name}")
        
        try:
            # Extract content
            print("  📄 Extracting content (text, tables)...")
            content = extractor.extract_with_pdfplumber(str(pdf_path))
            
            if not content or not content["full_text"].strip():
                print(f"  ⚠️ Failed to extract content, trying PyPDF2...")
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
                    "chunks": extractor.intelligent_chunking(text)
                }
            
            print(f"  ✅ Extracted {len(content['full_text'])} characters from {content['metadata']['num_pages']} pages")
            print(f"  📊 Created {len(content.get('chunks', []))} intelligent chunks")
            if content["tables"]:
                print(f"  📋 Found {len(content['tables'])} tables")
            
            # Generate questions
            print(f"  🤖 Generating {questions_per_doc} questions using {OLLAMA_MODEL}...")
            questions = generator.generate_questions_from_document(
                content, doc_id, num_questions=questions_per_doc
            )
            
            if questions:
                all_questions.extend(questions)
                doc_count += 1
                print(f"  ✅ Generated {len(questions)} validated questions")
            else:
                print(f"  ⚠️ No valid questions generated")
                failed_docs.append((pdf_path.name, "No valid questions generated"))
        
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {str(e)[:100]}")
            failed_docs.append((pdf_path.name, str(e)[:100]))
            continue
    
    # Analyze question distribution
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
            "question_statistics": question_stats
        },
        "questions": all_questions,
        "failed_documents": [{"filename": name, "reason": reason} for name, reason in failed_docs]
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE")
    print("="*80)
    print(f"Documents Processed: {doc_count}")
    print(f"Total Questions: {len(all_questions)}")
    print(f"Failed Documents: {len(failed_docs)}")
    print(f"Output saved to: {output_path}")
    print("="*80)
    
    # Print statistics
    print("\n📊 QUESTION STATISTICS:")
    print("-"*80)
    for key, value in question_stats.items():
        print(f"{key}: {value}")
    print("-"*80)
    
    # Print sample questions
    if all_questions:
        print("\n🔍 SAMPLE QUESTIONS:")
        print("-"*80)
        # Show examples of different types
        sample_types = ['multi-hop', 'cross-page', 'causal', 'single-hop']
        for q_type in sample_types:
            matching = [q for q in all_questions if q.get('question_type', '').startswith(q_type)]
            if matching:
                q = matching[0]
                print(f"\n[{q['question_type'].upper()}] Difficulty: {q['difficulty']}")
                print(f"Q: {q['question']}")
                print(f"A: {q['answer'][:150]}...")
                if 'reasoning_steps' in q and q['reasoning_steps']:
                    print(f"Reasoning: {q['reasoning_steps']}")
        print("-"*80)
    
    # Print failed documents if any
    if failed_docs:
        print("\n⚠️ FAILED DOCUMENTS:")
        print("-"*80)
        for name, reason in failed_docs:
            print(f"  • {name}: {reason}")
        print("-"*80)


def analyze_questions(questions: List[Dict]) -> Dict[str, Any]:
    """Analyze question distribution and statistics"""
    stats = {
        "by_type": {},
        "by_difficulty": {},
        "multi_hop_percentage": 0,
        "avg_answer_length": 0,
        "questions_with_evidence": 0
    }
    
    total = len(questions)
    if total == 0:
        return stats
    
    answer_lengths = []
    multi_hop_count = 0
    evidence_count = 0
    
    for q in questions:
        # Count by type
        q_type = q.get('question_type', 'unknown')
        stats['by_type'][q_type] = stats['by_type'].get(q_type, 0) + 1
        
        # Count by difficulty
        difficulty = q.get('difficulty', 'unknown')
        stats['by_difficulty'][difficulty] = stats['by_difficulty'].get(difficulty, 0) + 1
        
        # Multi-hop detection
        if 'multi-hop' in q_type or 'cross-page' in q_type or 'causal' in q_type:
            multi_hop_count += 1
        
        # Answer length
        if 'answer' in q:
            answer_lengths.append(len(q['answer']))
        
        # Evidence snippets
        if 'evidence_snippets' in q and q['evidence_snippets']:
            evidence_count += 1
    
    stats['multi_hop_percentage'] = round(multi_hop_count / total * 100, 2)
    stats['avg_answer_length'] = round(sum(answer_lengths) / len(answer_lengths)) if answer_lengths else 0
    stats['questions_with_evidence'] = evidence_count
    
    return stats


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "/home/olj3kor/praveen/RAG_work/DATASET/"
    OUTPUT_PATH = "/home/olj3kor/praveen/RAG_work/evaluation_questions_Qwen_72b.json"
    QUESTIONS_PER_DOC = 10  # Generate 8 questions per document
    
    # Verify Ollama is accessible
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        print(f"✅ Ollama is running at {OLLAMA_BASE_URL}\n")
    except Exception as e:
        print(f"❌ ERROR: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print(f"   Error: {e}")
        print("\nMake sure Ollama is running:")
        print("   docker ps | grep ollama")
        print("   OR: systemctl status ollama")
        exit(1)
    
    # Run generation
    process_dataset(DATASET_PATH, OUTPUT_PATH, QUESTIONS_PER_DOC)