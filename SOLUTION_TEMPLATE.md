# SMILES-2026 Hallucination Detection — Solution Report

## Reproducibility Instructions

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/kartsev-svg/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Solution

```bash
# Run the full pipeline (feature extraction, training, predictions)
python solution.py

# This generates:
# - results.json (evaluation metrics across folds)
# - predictions.csv (test set predictions)
```

### Important Notes

- **GPU**: The solution runs on CPU but is significantly faster with GPU (CUDA/MPS).
- **Data**: You must have `data/dataset.csv` and `data/test.csv` in the `data/` directory.
- **Runtime**: Expect ~5-15 minutes depending on hardware.
- **Reproducibility**: Random seed is fixed at `random_state=42` in `splitting.py`.

---

## Final Solution Description

### Modified Components

#### 1. `aggregation.py` — Feature Extraction

**What was changed:**
- **Before**: Used only the last real token from the final layer (896-dim).
- **After**: [Describe your approach]

**Rationale:**
- [Explain why you chose this aggregation strategy]

#### 2. `probe.py` — Binary Classifier

**What was changed:**
- **Before**: Single hidden layer (896 → 256 → 1).
- **After**: [Describe your architecture]

**Rationale:**
- [Explain design choices: depth, width, dropout, etc.]

#### 3. `splitting.py` — Train/Val/Test Split

**What was changed:**
- **Before**: Single stratified 70/15/15 split.
- **After**: [Describe your split strategy]

**Rationale:**
- [Explain why this split strategy is better]

### Final Approach Summary

[Provide a 2-3 paragraph summary of your complete solution strategy]

### Key Insights

1. **Feature Selection**: [What layer(s) and token positions matter most?]
2. **Architecture**: [What network structure worked best?]
3. **Training**: [Any special tricks (early stopping, learning rate, etc.)?]
4. **Validation**: [How did you validate improvements?]

---

## Experiments and Failed Attempts

### Experiment 1: [Description]

**What was tried:**
- [Technical details]

**Results:**
- Baseline accuracy: X%
- This approach accuracy: Y%

**Why it didn't work (or was discarded):**
- [Analysis and lessons learned]

---

### Experiment 2: [Description]

**What was tried:**
- [Technical details]

**Results:**
- Validation F1: X
- Test accuracy: Y%

**Why it didn't work (or was discarded):**
- [Analysis and lessons learned]

---

## Final Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | X% | Y% | Z% |
| F1-Score | X | Y | Z |
| AUROC | X | Y | Z |

---

## Conclusion

[1-2 paragraphs summarizing your solution, key achievements, and what would be next steps if you had more time]
