"""Inspect recursive chunks quality"""
import json
import random

chunks_file = "data/processed/chunks_recursive_512.json"

with open(chunks_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("CHUNK QUALITY INSPECTION")
print("=" * 80)
print(f"Total chunks: {data['metadata']['total_chunks']}")
print(f"Method: {data['metadata']['method']}")
print(f"Avg tokens: {data['metadata']['avg_chunk_tokens']:.1f}")
print("=" * 80)

# Show 3 random chunks
samples = random.sample(data['chunks'], 3)

for i, chunk in enumerate(samples, 1):
    print(f"\n{'='*80}")
    print(f"SAMPLE {i}/3")
    print(f"{'='*80}")
    print(f"Paper: {chunk['paper_filename'][:60]}")
    print(f"Chunk {chunk['chunk_id'] + 1}/{chunk['total_chunks_in_paper']}")
    print(f"Tokens: {chunk['chunk_tokens']} | Chars: {chunk['chunk_chars']}")
    print(f"\n{'-'*80}")
    print("CONTENT (first 400 chars):")
    print(f"{'-'*80}")
    print(chunk['chunk_text'][:400] + "...")
    print(f"{'-'*80}\n")