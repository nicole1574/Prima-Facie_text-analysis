
# Prima Facie Textual Untranslatability Analysis

This project investigates the untranslatability of embodied female experience in the English legal context, using the play **Prima Facie** as a case study. The analysis combines close reading and computational text analysis methods to reveal the linguistic and structural opposition between legal discourse and trauma narrative.

## Project Structure

- `data/`  
  Raw and processed texts of *Prima Facie* and any annotation files.

- `notebooks/`  
  Jupyter notebooks for exploratory analysis, visualization, and method prototyping.

- `src/`  
  Python scripts for text preprocessing, linguistic feature extraction, and quantitative analysis.

- `results/`  
  Output figures, tables, and reports generated from analysis.

- `docs/`  
  Research documentation, background, and theoretical framework.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords vader_lexicon
   ```

2. **Prepare the script:**
   - Place your full *Prima Facie* script in `data/prima_facie_script.txt` (with clear "PART ONE" and "PART TWO" markers).

3. **Run the main analysis:**
   ```bash
   python src/main.py data/prima_facie_script.txt --output results
   ```

4. **Check outputs:**
   - See visualizations and reports in the `results/` folder.

## Key Features

- **Section-aware analysis:** Compares Part One (lawyer identity) and Part Two (victim identity)
- **Linguistic diversity, legal term usage, syntactic complexity, and trauma markers** quantified and visualized.
- **Reproducible scripts and Jupyter notebooks** for transparent, extensible research.

## Research Questions

1. How does the language of legal discourse in Part One contrast with the trauma narrative in Part Two?
2. What are the computationally observable markers of untranslatability and narrative rupture?
3. How can hybrid methods enhance feminist legal-literary critique?

---

*For details on theory and methodology, see [`docs/theory.md`](docs/theory.md). For code and technical documentation, see [`docs/codebook.md`](docs/codebook.md).*
