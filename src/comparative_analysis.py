import os
import numpy as np
import matplotlib.pyplot as plt

def compare_parts(text_data, legal_terms,
                 calculate_lexical_diversity,
                 analyze_legal_terminology,
                 analyze_sentence_structure,
                 analyze_emotion_expression,
                 analyze_trauma_markers):
    results = {}
    # Part One
    results["part_one"] = {
        "lexical_diversity": calculate_lexical_diversity(text_data["part_one"]),
        "legal_terms": analyze_legal_terminology(text_data["part_one"], legal_terms),
        "sentence_structure": analyze_sentence_structure(text_data["part_one"]),
        "emotion": analyze_emotion_expression(text_data["part_one"]),
        "trauma_markers": analyze_trauma_markers(text_data["part_one"])
    }
    # Part Two
    results["part_two"] = {
        "lexical_diversity": calculate_lexical_diversity(text_data["part_two"]),
        "legal_terms": analyze_legal_terminology(text_data["part_two"], legal_terms),
        "sentence_structure": analyze_sentence_structure(text_data["part_two"]),
        "emotion": analyze_emotion_expression(text_data["part_two"]),
        "trauma_markers": analyze_trauma_markers(text_data["part_two"])
    }
    return results

def visualize_comparison(comparison_results, output_dir, legal_terms):
    os.makedirs(output_dir, exist_ok=True)
    parts = ["part_one", "part_two"]

    # Lexical Diversity
    plt.figure(figsize=(8, 5))
    ttr_values = [comparison_results[p]["lexical_diversity"]["ttr"] for p in parts]
    plt.bar(["Lawyer (Part One)", "Victim (Part Two)"], ttr_values)
    plt.title("Lexical Diversity (TTR)")
    plt.ylabel("Type-Token Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lexical_diversity.png"))
    plt.close()

    # Legal Terms (top 10)
    plt.figure(figsize=(10, 6))
    top_terms = sorted(
        [(term, comparison_results["part_one"]["legal_terms"][term]["frequency"])
         for term in legal_terms],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    terms = [t[0] for t in top_terms]
    p1_freq = [comparison_results["part_one"]["legal_terms"][t]["frequency"]*1000 for t in terms]
    p2_freq = [comparison_results["part_two"]["legal_terms"][t]["frequency"]*1000 for t in terms]
    x = np.arange(len(terms))
    width = 0.35
    plt.bar(x - width/2, p1_freq, width, label="Lawyer (Part One)")
    plt.bar(x + width/2, p2_freq, width, label="Victim (Part Two)")
    plt.xlabel("Legal Term")
    plt.ylabel("Frequency (per 1000 words)")
    plt.title("Legal Terms Usage")
    plt.xticks(x, terms, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "legal_terms.png"))
    plt.close()

    # Sentence Structure
    plt.figure(figsize=(8, 5))
    metrics = ["avg_length", "complex_sentence_rate", "fragment_rate"]
    labels = ["Avg Sentence Length", "Complex Sentences", "Fragments"]
    p1_values = [comparison_results["part_one"]["sentence_structure"][m] for m in metrics]
    p2_values = [comparison_results["part_two"]["sentence_structure"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, p1_values, width, label="Lawyer (Part One)")
    plt.bar(x + width/2, p2_values, width, label="Victim (Part Two)")
    plt.xlabel("Syntactic Metric")
    plt.ylabel("Value")
    plt.title("Sentence Structure Comparison")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sentence_structure.png"))
    plt.close()

    # Emotion
    plt.figure(figsize=(8, 5))
    metrics = ["avg_sentiment", "sentiment_variation", "emotional_intensity"]
    labels = ["Avg Sentiment", "Variation", "Intensity"]
    p1_values = [comparison_results["part_one"]["emotion"][m] for m in metrics]
    p2_values = [comparison_results["part_two"]["emotion"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, p1_values, width, label="Lawyer (Part One)")
    plt.bar(x + width/2, p2_values, width, label="Victim (Part Two)")
    plt.xlabel("Emotion Metric")
    plt.ylabel("Value")
    plt.title("Emotion Expression Comparison")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "emotion_expression.png"))
    plt.close()

    # Trauma Markers
    plt.figure(figsize=(10, 5))
    metrics = ["tense_shift_rate", "repetition_rate", "sensory_rate", "disruption_rate"]
    labels = ["Tense Shift", "Repetition", "Sensory", "Disruption"]
    p1_values = [comparison_results["part_one"]["trauma_markers"][m] for m in metrics]
    p2_values = [comparison_results["part_two"]["trauma_markers"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, p1_values, width, label="Lawyer (Part One)")
    plt.bar(x + width/2, p2_values, width, label="Victim (Part Two)")
    plt.xlabel("Trauma Marker")
    plt.ylabel("Rate")
    plt.title("Trauma Markers Comparison")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trauma_markers.png"))
    plt.close()
