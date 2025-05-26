import os
import argparse
from datetime import datetime
import pandas as pd
from prima_facie_analysis import preprocess_script
from linguistic_analysis import (
    calculate_lexical_diversity, analyze_legal_terminology,
    analyze_sentence_structure, analyze_emotion_expression,
    analyze_trauma_markers
)
from comparative_analysis import compare_parts, visualize_comparison

legal_terms = [
    "evidence", "testimony", "witness", "cross-examination", "prosecution",
    "defense", "objection", "alleged", "reasonable doubt", "burden of proof",
    "victim", "accused", "court", "judge", "jury", "beyond reasonable doubt",
    "consent", "credibility", "hearsay", "sworn", "testimony", "verdict",
    "examination-in-chief", "precedent", "prima facie", "acquittal",
    "adversarial", "complainant", "counsel", "defendant", "jurisdiction"
]

def run_analysis(file_path, output_dir):
    print(f"Starting Prima Facie text analysis...")
    start_time = datetime.now()
    os.makedirs(output_dir, exist_ok=True)
    # Step 1: Preprocess
    print("1. Preprocessing script...")
    text_data = preprocess_script(file_path)
    # Step 2: Compare Part One and Part Two
    print("2. Comparing text segments...")
    comparison_results = compare_parts(
        text_data, legal_terms,
        calculate_lexical_diversity, analyze_legal_terminology,
        analyze_sentence_structure, analyze_emotion_expression,
        analyze_trauma_markers
    )
    # Step 3: Visualization
    print("3. Visualizing results...")
    visualize_comparison(comparison_results, output_dir, legal_terms)
    # Step 4: Report
    generate_report(comparison_results, text_data, output_dir)
    print(f"Analysis complete! Time used: {datetime.now() - start_time}")
    print(f"Results saved to {output_dir}")

def generate_report(comparison_results, text_data, output_dir):
    part_one_words = len(text_data["part_one"].split())
    part_two_words = len(text_data["part_two"].split())
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.write("Prima Facie Text Analysis Report\n")
        f.write("="*40 + "\n\n")
        f.write("1. Basic Stats\n")
        f.write(f"Part One (Lawyer) words: {part_one_words}\n")
        f.write(f"Part Two (Victim) words: {part_two_words}\n")
        f.write(f"Scene count: {len(text_data['scenes'])}\n\n")
        f.write("2. Lexical Diversity\n")
        f.write(f"Part One TTR: {comparison_results['part_one']['lexical_diversity']['ttr']:.4f}\n")
        f.write(f"Part Two TTR: {comparison_results['part_two']['lexical_diversity']['ttr']:.4f}\n\n")
        f.write("3. Syntactic Structure\n")
        f.write(f"Part One Avg Sentence Length: {comparison_results['part_one']['sentence_structure']['avg_length']:.2f}\n")
        f.write(f"Part Two Avg Sentence Length: {comparison_results['part_two']['sentence_structure']['avg_length']:.2f}\n")
        f.write(f"Part One Complex Sentence Rate: {comparison_results['part_one']['sentence_structure']['complex_sentence_rate']:.2f}\n")
        f.write(f"Part Two Complex Sentence Rate: {comparison_results['part_two']['sentence_structure']['complex_sentence_rate']:.2f}\n")
        f.write(f"Part One Fragment Rate: {comparison_results['part_one']['sentence_structure']['fragment_rate']:.2f}\n")
        f.write(f"Part Two Fragment Rate: {comparison_results['part_two']['sentence_structure']['fragment_rate']:.2f}\n\n")
        f.write("4. Emotion\n")
        f.write(f"Part One Avg Sentiment: {comparison_results['part_one']['emotion']['avg_sentiment']:.2f}\n")
        f.write(f"Part Two Avg Sentiment: {comparison_results['part_two']['emotion']['avg_sentiment']:.2f}\n")
        f.write(f"Part One Sentiment Variation: {comparison_results['part_one']['emotion']['sentiment_variation']:.2f}\n")
        f.write(f"Part Two Sentiment Variation: {comparison_results['part_two']['emotion']['sentiment_variation']:.2f}\n\n")
        f.write("5. Trauma Markers\n")
        f.write(f"Part One Tense Shift Rate: {comparison_results['part_one']['trauma_markers']['tense_shift_rate']:.2f}\n")
        f.write(f"Part Two Tense Shift Rate: {comparison_results['part_two']['trauma_markers']['tense_shift_rate']:.2f}\n")
        f.write(f"Part One Repetition Rate: {comparison_results['part_one']['trauma_markers']['repetition_rate']:.2f}\n")
        f.write(f"Part Two Repetition Rate: {comparison_results['part_two']['trauma_markers']['repetition_rate']:.2f}\n")
        f.write(f"Part One Sensory Rate: {comparison_results['part_one']['trauma_markers']['sensory_rate']:.2f}\n")
        f.write(f"Part Two Sensory Rate: {comparison_results['part_two']['trauma_markers']['sensory_rate']:.2f}\n")
        f.write(f"Part One Disruption Rate: {comparison_results['part_one']['trauma_markers']['disruption_rate']:.2f}\n")
        f.write(f"Part Two Disruption Rate: {comparison_results['part_two']['trauma_markers']['disruption_rate']:.2f}\n\n")
        f.write("6. Top 5 Legal Terms (per 1000 words)\n")
        p1_terms = sorted(
            [(term, comparison_results["part_one"]["legal_terms"][term]["frequency"]*1000)
             for term in legal_terms], key=lambda x: x[1], reverse=True)[:5]
        p2_terms = sorted(
            [(term, comparison_results["part_two"]["legal_terms"][term]["frequency"]*1000)
             for term in legal_terms], key=lambda x: x[1], reverse=True)[:5]
        f.write("Part One:\n")
        for term, freq in p1_terms:
            f.write(f"  - {term}: {freq:.2f}\n")
        f.write("Part Two:\n")
        for term, freq in p2_terms:
            f.write(f"  - {term}: {freq:.2f}\n")
    # Also save detailed term data for further research
    p1_legal = {term: comparison_results["part_one"]["legal_terms"][term]["frequency"]*1000 for term in legal_terms}
    p2_legal = {term: comparison_results["part_two"]["legal_terms"][term]["frequency"]*1000 for term in legal_terms}
    legal_df = pd.DataFrame({
        "Term": legal_terms,
        "Part_One": [p1_legal[term] for term in legal_terms],
        "Part_Two": [p2_legal[term] for term in legal_terms]
    })
    legal_df.to_csv(os.path.join(output_dir, "legal_terms_data.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Prima Facie text for untranslatability research')
    parser.add_argument('file_path', help='Path to the Prima Facie script text file')
    parser.add_argument('--output', default='results', help='Output directory for results')
    args = parser.parse_args()
    run_analysis(args.file_path, args.output)