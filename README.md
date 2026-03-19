# dimensionality-reduction-comparison

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Comparison of dimensionality reduction techniques on the MNIST dataset, across two exercises: **training efficiency** and **2D visual analysis**.

---

## Aim of this project

This project evaluates how different dimensionality reduction methods affect a machine learning pipeline, split into two complementary exercises:

**Exercise 1 — Training Efficiency:** measures reduction time, model training time, and classification accuracy (Logistic Regression) for each technique applied to the full pipeline.

**Exercise 2 — 2D Visualization:** projects the data into 2D space and compares how well each technique separates the digit classes visually.

> **Note on subsets:** Kernel PCA, LLE and t-SNE are computationally expensive (O(n²)–O(n³) complexity), so they run on a subset of 10,000 samples. Their results are not directly comparable to full-dataset techniques.

---

## Techniques compared

| Technique | Exercise 1 | Exercise 2 | Data used |
|---|---|---|---|
| PCA | ✅ | ✅ | Full dataset / 10k subset |
| Randomized PCA | ✅ | — | Full dataset |
| Incremental PCA | ✅ | — | Full dataset |
| Kernel PCA | ✅ | ✅ | Subset (10k) |
| LLE | ✅ | ✅ | Subset (10k) |
| t-SNE | — | ✅ | Subset (10k) |

---

## Project Structure

```
dimensionality-reduction-comparison/
│
├── src/
│   └── main.py          # Full pipeline (Exercise 1 + Exercise 2 + summary)
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/moralesgomez-dev/dimensionality-reduction-comparison.git
cd dimensionality-reduction-comparison
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.\.venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python src/main.py
```

---

## Pipeline Overview

1. Load MNIST via `fetch_openml` (70,000 samples, 784 features)
2. Data inspection: sample digit + class distribution
3. Train/test split: 60,000 / 10,000
4. StandardScaler preprocessing (required for distance-based methods)
5. **Exercise 1:** baseline + 5 techniques → reduction time, training time, accuracy
6. **Exercise 2:** 4 techniques → 2D scatter plots colored by digit class
7. **Summary report:** consolidated DataFrame for both exercises

---

## Summary Report (console output)

At the end of the run, a report is printed with:

- `reduction_time_s` — time to apply the technique
- `train_time_s` — time to train Logistic Regression on reduced data
- `accuracy` — test set accuracy
- `n_components` — output dimensions
- `data_used` — full dataset or subset
- `total_time_s` — combined time
- `acc_vs_baseline` — accuracy delta vs no reduction

Automatically highlights: best accuracy, fastest technique, best accuracy/time ratio.

---

## Contributing

1. Fork the project
2. Create your branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## Author

**AlejandroMoralesGomezDev**
- GitHub: [moralesgomez-dev](https://github.com/moralesgomez-dev)
- Kaggle: [moralesgomez](https://www.kaggle.com/moralesgomez)

---

## License

MIT License — see [LICENSE](LICENSE) for details.