"""
Question Quality Verification Tool
Review and validate generated RAG evaluation questions
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

class QuestionQualityVerifier:
    """Tools for manual and automated quality verification of generated questions"""
    
    def __init__(self, questions_file: str, pdf_dir: str = None):
        """
        Initialize verifier
        Args:
            questions_file: Path to JSON file with generated questions
            pdf_dir: Optional path to PDF directory for context verification
        """
        self.questions_file = questions_file
        self.pdf_dir = pdf_dir
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.questions = self.data.get('questions', [])
        self.stats = self.data.get('dataset_info', {}).get('question_statistics', {})
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            "total_questions": len(self.questions),
            "quality_checks": {},
            "warnings": [],
            "recommendations": []
        }
        
        # Check 1: Question Type Distribution
        type_dist = Counter(q.get('question_type', 'unknown') for q in self.questions)
        report['quality_checks']['type_distribution'] = dict(type_dist)
        
        # Verify multi-hop coverage
        multi_hop_types = ['multi-hop', 'cross-page', 'causal', 'temporal', 'comparison']
        multi_hop_count = sum(count for qtype, count in type_dist.items() 
                              if any(mh in qtype for mh in multi_hop_types))
        multi_hop_pct = (multi_hop_count / len(self.questions)) * 100 if self.questions else 0
        
        report['quality_checks']['multi_hop_percentage'] = round(multi_hop_pct, 2)
        
        if multi_hop_pct < 30:
            report['warnings'].append(
                f"⚠️ Low multi-hop question ratio ({multi_hop_pct:.1f}%). Target: 30%+"
            )
        
        # Check 2: Difficulty Distribution
        difficulty_dist = Counter(q.get('difficulty', 'unknown') for q in self.questions)
        report['quality_checks']['difficulty_distribution'] = dict(difficulty_dist)
        
        # Check 3: Answer Quality
        answer_issues = []
        for q in self.questions:
            answer = q.get('answer', '')
            if len(answer) < 10:
                answer_issues.append(f"Question {q.get('id')}: Answer too short")
            elif len(answer) > 500:
                answer_issues.append(f"Question {q.get('id')}: Answer too long")
        
        if answer_issues:
            report['warnings'].extend(answer_issues[:5])  # Show first 5
            if len(answer_issues) > 5:
                report['warnings'].append(f"... and {len(answer_issues) - 5} more answer issues")
        
        # Check 4: Evidence Coverage
        questions_with_evidence = sum(1 for q in self.questions 
                                     if 'evidence_snippets' in q and q['evidence_snippets'])
        evidence_pct = (questions_with_evidence / len(self.questions)) * 100 if self.questions else 0
        report['quality_checks']['evidence_coverage'] = f"{evidence_pct:.1f}%"
        
        if evidence_pct < 50:
            report['warnings'].append(
                f"⚠️ Low evidence coverage ({evidence_pct:.1f}%). Many questions lack evidence snippets."
            )
        
        # Check 5: Reasoning Steps (for complex questions)
        reasoning_count = sum(1 for q in self.questions 
                            if 'reasoning_steps' in q and q['reasoning_steps'])
        report['quality_checks']['questions_with_reasoning'] = reasoning_count
        
        # Check 6: Duplicate Detection
        question_texts = [q['question'].lower().strip() for q in self.questions]
        duplicates = [q for q in question_texts if question_texts.count(q) > 1]
        if duplicates:
            report['warnings'].append(
                f"⚠️ Found {len(set(duplicates))} potential duplicate questions"
            )
        
        # Check 7: Page Coverage
        page_refs = [q.get('page_reference', '') for q in self.questions]
        multi_page_count = sum(1 for ref in page_refs if 'Multiple' in ref or '-' in ref)
        report['quality_checks']['cross_page_questions'] = multi_page_count
        
        # Generate Recommendations
        if multi_hop_pct < 30:
            report['recommendations'].append(
                "Increase multi-hop and reasoning questions for better RAG testing"
            )
        
        if evidence_pct < 70:
            report['recommendations'].append(
                "Add more evidence snippets to validate answer grounding"
            )
        
        easy_pct = (difficulty_dist.get('easy', 0) / len(self.questions)) * 100
        if easy_pct > 30:
            report['recommendations'].append(
                f"Too many easy questions ({easy_pct:.1f}%). Increase medium/hard questions."
            )
        
        return report
    
    def sample_questions_by_category(self) -> Dict[str, List[Dict]]:
        """Get sample questions from each category for manual review"""
        samples = {}
        
        categories = [
            'single-hop', 'multi-hop', 'cross-page', 'numerical',
            'causal', 'comparison', 'temporal', 'definition', 'list'
        ]
        
        for category in categories:
            matching = [q for q in self.questions 
                       if category in q.get('question_type', '').lower()]
            if matching:
                samples[category] = matching[:3]  # Take 3 samples
        
        return samples
    
    def identify_problematic_questions(self) -> List[Dict[str, Any]]:
        """Identify questions that may need review"""
        problematic = []
        
        for q in self.questions:
            issues = []
            
            # Check answer length
            answer = q.get('answer', '')
            if len(answer) < 20:
                issues.append("Very short answer")
            elif len(answer) > 400:
                issues.append("Very long answer")
            
            # Check question clarity
            question_text = q.get('question', '')
            if question_text.count('?') > 1:
                issues.append("Multiple question marks")
            
            # Check for vague terms
            vague_terms = ['something', 'anything', 'somehow', 'maybe', 'possibly']
            if any(term in question_text.lower() for term in vague_terms):
                issues.append("Contains vague language")
            
            # Check missing fields
            if not q.get('evidence_snippets'):
                if 'multi-hop' in q.get('question_type', '') or 'cross-page' in q.get('question_type', ''):
                    issues.append("Complex question missing evidence")
            
            if issues:
                problematic.append({
                    'question_id': q.get('id'),
                    'question': question_text[:100],
                    'issues': issues,
                    'type': q.get('question_type'),
                    'difficulty': q.get('difficulty')
                })
        
        return problematic
    
    def export_review_sheet(self, output_file: str):
        """Export questions to a reviewable format (TSV for spreadsheet)"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("ID\tType\tDifficulty\tQuestion\tAnswer\tEvidence\tReviewer_Notes\tApproved\n")
            
            for q in self.questions:
                evidence = '; '.join(q.get('evidence_snippets', []))[:200]
                
                row = [
                    q.get('id', ''),
                    q.get('question_type', ''),
                    q.get('difficulty', ''),
                    q.get('question', '').replace('\t', ' ').replace('\n', ' '),
                    q.get('answer', '')[:200].replace('\t', ' ').replace('\n', ' '),
                    evidence.replace('\t', ' ').replace('\n', ' '),
                    '',  # Reviewer notes
                    ''   # Approved (Y/N)
                ]
                f.write('\t'.join(row) + '\n')
        
        print(f"✅ Review sheet exported to: {output_file}")
        print("   Open in Excel/Google Sheets for manual review")
    
    def check_answer_grounding(self, sample_size: int = 10) -> List[Dict]:
        """
        Sample questions and check if answers are properly grounded
        Returns questions that may need verification
        """
        if not self.pdf_dir:
            print("⚠️ PDF directory not provided. Skipping grounding check.")
            return []
        
        import pdfplumber
        import random
        
        # Sample questions
        sample = random.sample(self.questions, min(sample_size, len(self.questions)))
        
        needs_review = []
        
        for q in sample:
            doc_name = q.get('source_document', '')
            pdf_path = Path(self.pdf_dir) / doc_name
            
            if not pdf_path.exists():
                needs_review.append({
                    'question_id': q['id'],
                    'issue': 'Source document not found',
                    'question': q['question'][:100]
                })
                continue
            
            # Extract text from PDF
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    full_text = ' '.join(page.extract_text() or '' for page in pdf.pages)
                
                # Check if answer terms appear in document
                answer = q.get('answer', '').lower()
                answer_words = set(w for w in answer.split() if len(w) > 3)
                
                doc_words = set(full_text.lower().split())
                
                # At least 50% of significant answer words should be in document
                overlap = len(answer_words.intersection(doc_words))
                if len(answer_words) > 0 and overlap / len(answer_words) < 0.5:
                    needs_review.append({
                        'question_id': q['id'],
                        'issue': f'Low answer grounding ({overlap}/{len(answer_words)} words found)',
                        'question': q['question'][:100],
                        'answer': answer[:100]
                    })
            
            except Exception as e:
                needs_review.append({
                    'question_id': q['id'],
                    'issue': f'Error checking grounding: {str(e)[:50]}',
                    'question': q['question'][:100]
                })
        
        return needs_review
    
    def print_quality_summary(self):
        """Print comprehensive quality summary"""
        report = self.generate_quality_report()
        
        print("="*80)
        print("QUESTION QUALITY VERIFICATION REPORT")
        print("="*80)
        print(f"Total Questions: {report['total_questions']}")
        print()
        
        print("📊 QUALITY CHECKS:")
        print("-"*80)
        for check, value in report['quality_checks'].items():
            print(f"  {check}: {value}")
        print()
        
        if report['warnings']:
            print("⚠️  WARNINGS:")
            print("-"*80)
            for warning in report['warnings']:
                print(f"  {warning}")
            print()
        
        if report['recommendations']:
            print("💡 RECOMMENDATIONS:")
            print("-"*80)
            for rec in report['recommendations']:
                print(f"  • {rec}")
            print()
        
        print("="*80)
    
    def interactive_review(self):
        """Interactive CLI for reviewing questions"""
        print("\n🔍 INTERACTIVE QUESTION REVIEW")
        print("="*80)
        
        samples = self.sample_questions_by_category()
        
        for category, questions in samples.items():
            print(f"\n📂 Category: {category.upper()}")
            print("-"*80)
            
            for idx, q in enumerate(questions, 1):
                print(f"\n[{idx}] ID: {q.get('id')}")
                print(f"Difficulty: {q.get('difficulty')}")
                print(f"Q: {q.get('question')}")
                print(f"A: {q.get('answer')}")
                
                if q.get('reasoning_steps'):
                    print(f"Reasoning: {q.get('reasoning_steps')}")
                
                if q.get('evidence_snippets'):
                    print(f"Evidence: {q.get('evidence_snippets')[:2]}")
                
                print("-"*80)
        
        print(f"\n✅ Reviewed {sum(len(qs) for qs in samples.values())} sample questions")


def main():
    """Main verification workflow"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_questions.py <questions_json_file> [pdf_directory]")
        print("\nExample:")
        print("  python verify_questions.py evaluation_questions_improved.json /path/to/PDFs/")
        sys.exit(1)
    
    questions_file = sys.argv[1]
    pdf_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(questions_file):
        print(f"❌ Error: File not found: {questions_file}")
        sys.exit(1)
    
    verifier = QuestionQualityVerifier(questions_file, pdf_dir)
    
    # Generate and print quality report
    verifier.print_quality_summary()
    
    # Identify problematic questions
    print("\n🔴 POTENTIALLY PROBLEMATIC QUESTIONS:")
    print("="*80)
    problematic = verifier.identify_problematic_questions()
    
    if problematic:
        for item in problematic[:10]:  # Show first 10
            print(f"\nID: {item['question_id']} | Type: {item['type']} | Difficulty: {item['difficulty']}")
            print(f"Issues: {', '.join(item['issues'])}")
            print(f"Q: {item['question']}")
        
        if len(problematic) > 10:
            print(f"\n... and {len(problematic) - 10} more problematic questions")
    else:
        print("✅ No major issues found!")
    
    print("\n" + "="*80)
    
    # Check answer grounding if PDFs available
    if pdf_dir:
        print("\n📋 CHECKING ANSWER GROUNDING (sample):")
        print("="*80)
        grounding_issues = verifier.check_answer_grounding(sample_size=5)
        
        if grounding_issues:
            for issue in grounding_issues:
                print(f"\n⚠️ {issue['question_id']}: {issue['issue']}")
                print(f"   Q: {issue['question']}")
        else:
            print("✅ Sample questions appear well-grounded")
        print("="*80)
    
    # Export review sheet
    review_file = questions_file.replace('.json', '_review_sheet.tsv')
    verifier.export_review_sheet(review_file)
    
    # Interactive review option
    print("\n" + "="*80)
    response = input("\nWould you like to see sample questions for manual review? (y/n): ")
    if response.lower() == 'y':
        verifier.interactive_review()
    
    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()