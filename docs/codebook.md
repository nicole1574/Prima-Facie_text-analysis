# Codebook: Prima Facie Text Analysis

## File Overview

- `src/prima_facie_analysis.py`  
  - Reads and segments script by PART ONE / PART TWO, extracts scenes.
- `src/linguistic_analysis.py`  
  - Functions for lexical diversity, legal terms frequency, sentence structure, emotion, and trauma markers.
- `src/comparative_analysis.py`  
  - Compares features between parts, generates visualizations.
- `src/main.py`  
  - Main pipeline integrating all modules, generates report and figures.

## Data Preparation

- **Script file:**  
  - Must be plain text, with clear "PART ONE" and "PART TWO" markers.
  - Scene headers as "Scene 1", "Scene 2", etc. (optional for fine-grained analysis).

## Running Analysis

```bash
python src/main.py data/prima_facie_script.txt --output results
```

## Output

- `results/analysis_report.txt`: Key findings.
- `results/*.png`: Visualizations (lexical diversity, legal terms, etc).
- `results/legal_terms_data.csv`: Term frequencies.

## Customization

- **Legal terms list:**  
  - Update `legal_terms` in `src/main.py` for domain-specific focus.
- **Scene-level analysis:**  
  - Expand code to analyze individual scenes if needed.
