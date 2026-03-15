"""Quick quality check on extracted text"""
import json

with open('data/processed/extracted_text.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

print("=" * 60)
print("EXTRACTION QUALITY REPORT")
print("=" * 60)

for i, paper in enumerate(results, 1):
    text_length = len(paper['text'])
    status = paper['status']

    # Simple quality heuristic
    if status == 'success' and text_length > 5000:
        quality = "✓ GOOD"
    elif status == 'success' and text_length > 1000:
        quality = "⚠ OK (short)"
    else:
        quality = "✗ POOR"

    print(f"\n{i}. {paper['filename'][:50]}...")
    print(f"   Status: {status}")
    print(f"   Length: {text_length:,} chars")
    print(f"   Pages: {paper['num_pages']}")
    print(f"   Quality: {quality}")

successful = sum(1 for p in results if p['status'] == 'success')
good_quality = sum(1 for p in results if p['status'] == 'success' and len(p['text']) > 5000)

print("\n" + "=" * 60)
print(f"Summary:")
print(f"  Total: {len(results)}")
print(f"  Extracted successfully: {successful}")
print(f"  Good quality: {good_quality}")
print("=" * 60)