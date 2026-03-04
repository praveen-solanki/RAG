# import json
# import sys
# from collections import Counter

# # -------- EXPECTED DOCUMENTS --------
# EXPECTED_DOCS = {
#     r"TroubleShooting Manual_IndraCs upto MPx18Vrs [Ed05].pdf",
#     r"TroubleShooting Manual_IndraDrive upto MPx08 [Ed 08].pdf",
# }
# # EXPECTED_DOCS = {
# #     "Technical Report on Operating System Tracing Interface.pdf",
# #     "General Specification of Transformers.pdf",
# #     "Requirements on Operating System Interface.pdf",
# #     "Adaptive Platform Machine Configuration.pdf",
# #     "Explanation of Service-Oriented Vehicle Diagnostics.pdf",
# #     "Specification of Raw Data Stream.pdf",
# #     "Utilization of Crypto Services.pdf",
# #     "Explanation of Sensor Interfaces.pdf",
# #     "Specification of Firewall for Adaptive Platform.pdf",
# #     "AUTOSAR Layered Software Architecture.pdf",
# # }


# def main(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     if "questions" not in data:
#         print("❌ JSON does not contain 'questions' key.")
#         return

#     questions = data["questions"]

#     # Extract documents
#     docs = [q["source_document"] for q in questions if "source_document" in q]

#     doc_counter = Counter(docs)
#     unique_docs = set(docs)

#     print("\n========== DOCUMENT CHECK ==========\n")
#     print(f"Total Questions: {len(questions)}")
#     print(f"Unique Documents Found: {len(unique_docs)}\n")

#     print("📊 Questions per document:")
#     for doc, count in sorted(doc_counter.items()):
#         print(f"  - {doc}: {count}")

#     print("\n-------------------------------------")

#     extra_docs = unique_docs - EXPECTED_DOCS
#     missing_docs = EXPECTED_DOCS - unique_docs

#     if not extra_docs:
#         print("\n✅ No unexpected documents found.")
#     else:
#         print("\n⚠️ Unexpected documents found:")
#         for doc in sorted(extra_docs):
#             print("  -", doc)

#     if not missing_docs:
#         print("\n✅ No expected documents missing.")
#     else:
#         print("\n⚠️ Expected documents missing:")
#         for doc in sorted(missing_docs):
#             print("  -", doc)

#     print("\n=====================================\n")


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python check_documents_in_json.py <json_file>")
#         sys.exit(1)

#     main(sys.argv[1])



import json

# INPUT_FILE  = "/home/olj3kor/praveen/first.json"
INPUT_FILE  = "/home/olj3kor/praveen/RAG_work/evaluation_questions_Qwen_72b.json"
OUTPUT_FILE = "retrieval_eval_2.json"

KEEP_FIELDS = [
    "id",
    "question",
    "source_document",
    "answer",
    "evidence_snippets",
    "page_reference",
    "difficulty",
    "question_type",
]

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

cleaned_questions = []
for q in data.get("questions", []):
    cleaned_questions.append({k: q[k] for k in KEEP_FIELDS if k in q})

output = {
    "dataset_info": {
        "total_questions": len(cleaned_questions),
        "source_file": INPUT_FILE,
    },
    "questions": cleaned_questions,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"Done. {len(cleaned_questions)} questions saved to {OUTPUT_FILE}")