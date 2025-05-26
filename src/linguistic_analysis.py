import re
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def calculate_lexical_diversity(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    if not words:
        return {"ttr": 0, "unique_words": 0, "total_words": 0}
    unique_words = set(words)
    return {"ttr": len(unique_words)/len(words), "unique_words": len(unique_words), "total_words": len(words)}

def analyze_legal_terminology(text, legal_terms):
    text_lower = text.lower()
    total_words = len([w for w in word_tokenize(text_lower) if w.isalpha()])
    results = {}
    for term in legal_terms:
        count = text_lower.count(term.lower())
        results[term] = {"count": count, "frequency": count/total_words if total_words > 0 else 0}
    return results

def analyze_sentence_structure(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return {"avg_length": 0, "complex_sentence_rate": 0, "fragment_rate": 0}
    avg_length = sum(len(sent) for sent in sentences) / len(sentences)
    complex_sentences = sum(sum(1 for token in sent if token.pos_ == "VERB") > 1 for sent in sentences)
    fragments = sum(not any(token.pos_ == "VERB" for token in sent) for sent in sentences)
    return {
        "avg_length": avg_length,
        "complex_sentence_rate": complex_sentences/len(sentences),
        "fragment_rate": fragments/len(sentences)
    }

def analyze_emotion_expression(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return {"avg_sentiment": 0, "sentiment_variation": 0, "emotional_intensity": 0}
    sentiment_scores = [sia.polarity_scores(s)["compound"] for s in sentences]
    avg_sentiment = np.mean(sentiment_scores)
    sentiment_variation = np.std(sentiment_scores)
    emotional_intensity = np.mean([abs(s) for s in sentiment_scores])
    return {
        "avg_sentiment": avg_sentiment,
        "sentiment_variation": sentiment_variation,
        "emotional_intensity": emotional_intensity
    }

def analyze_trauma_markers(text):
    doc = nlp(text)
    tense_shifts = 0
    for sent in doc.sents:
        tenses = set()
        for token in sent:
            if token.pos_ == "VERB":
                if token.tag_ in ["VBD", "VBN"]:
                    tenses.add("past")
                elif token.tag_ in ["VBZ", "VBP", "VB"]:
                    tenses.add("present")
        if len(tenses) > 1:
            tense_shifts += 1
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    word_counts = Counter(words)
    repetitions = sum(1 for word, count in word_counts.items() if count > 3 and word not in stop_words)
    sensory_words = ["see", "hear", "feel", "smell", "taste", "touch",
                     "saw", "heard", "felt", "body", "pain", "numb"]
    sensory_count = sum(text.lower().count(word) for word in sensory_words)
    ellipses = len(re.findall(r'\.{3}|…', text))
    dashes = len(re.findall(r'—|--', text))
    total_words = len([w for w in words if w not in stop_words])
    return {
        "tense_shifts": tense_shifts,
        "tense_shift_rate": tense_shifts / len(list(doc.sents)) if doc.sents else 0,
        "repetition_count": repetitions,
        "repetition_rate": repetitions / total_words if total_words > 0 else 0,
        "sensory_count": sensory_count,
        "sensory_rate": sensory_count / total_words if total_words > 0 else 0,
        "disruption_markers": ellipses + dashes,
        "disruption_rate": (ellipses + dashes) / len(list(doc.sents)) if doc.sents else 0
    }
