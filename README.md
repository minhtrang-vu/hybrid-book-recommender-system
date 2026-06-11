# 📚 NLP-Enhanced Book Discovery System Using Hybrid Recommendation

> **Course Project** — Business Analytics, K62 DB | Foreign Trade University – VJCC Institute  
> **April 2026**

---

## 📖 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [NLP Pipeline](#nlp-pipeline)
- [Recommender System](#recommender-system)
- [Evaluation & Results](#evaluation--results)
- [Prototype](#prototype)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## Overview

Traditional book recommendation systems on platforms like Goodreads rely heavily on historical ratings and popularity signals, often producing repetitive or generic suggestions. This project proposes an **NLP-Enhanced Book Discovery System** that enables users to discover books through **natural language queries** — expressing nuanced preferences such as emotional tone, narrative tropes, and contextual reading needs.

The system integrates:
- **Natural Language Processing (NLP)** to extract structured preference signals from unstructured queries
- **Content-based filtering** leveraging dual semantic embeddings of book descriptions and user reviews
- **Collaborative filtering** combining Bayesian ratings, interaction signals, and sentiment scores
- **Grid Search optimization** across 6 parameter configurations to identify the best-performing blend

### Key Results (Best Config — `D_content_heavy`)

| Metric | Validation | Test |
|---|---|---|
| NDCG@5 | 0.5160 | 0.4992 |
| Trope-Precision@5 | 0.3354 | 0.3604 |
| Catalog Coverage | 70.80% | 70.20% |
| Novelty | 96.92% | 97.96% |
| **Composite Score** | **0.5586** | **0.5601** |

> ✅ Minimal delta between VAL and TEST (Δ Composite = +0.0015), confirming stable generalization with no overfitting.

---

## System Architecture

```
User Query
    │
    ▼  Phase 1 — NLP Intent Extraction (extract_intent_final)
    ├─ Text normalization + synonym expansion
    ├─ spaCy tokenization, lemmatization, POS tagging
    ├─ Negation detection (dependency tree scope)
    ├─ Bigram-first keyword extraction
    ├─ Layer 1: keyword dict matching (TROPE_KEYWORDS, TONE_KEYWORDS)
    └─ Layer 2: zero-shot cosine similarity (threshold ≥ 0.45)
         │
    build_query_text(intent) → enriched query string
         │
    ┌────┴──────────────────────────────────────┐
    │                                           │
    ▼  Phase 2A — Content Score (w=0.75)        ▼  Phase 2B — Collab Score (w=0.25)
    process_query() → L2-norm (1, 384)          0.5 × bayesian_score
    compute_similarity():                       0.3 × interaction_score
      0.7 × desc_emb + 0.3 × review_emb        0.2 × sentiment_score
    + genre/trope boost (+0.05–0.15)                 ↓
    normalize [0, 1]                        (known user: 0.6 CF + 0.4 global)
    content_score                               collab_score
    │                                           │
    └──────────────────┬────────────────────────┘
                       ▼  Phase 3 — Final Blend
              0.75 × content + 0.25 × collab
                       │
              − 0.10 × neg_similarity × n_neg_kws
                       │
              clip [0,1] → exclude_read → sort → top-N
```

---

## Dataset

Three data sources are used:

| File | Description | Size |
|---|---|---|
| `6000_bookssamples.csv` | Book metadata: title, author, description, genre, rating, totalratings, pages, bookformat | ~6,000 books (filtered from 100,000) |
| `review_clean.csv` | User reviews crawled from Goodreads via GoogleAIStudio | ~75,801 clean English reviews |
| `interactions.csv` | Simulated interaction events: click, save, search, purchase, share | 450,020 interactions, 33,643 users |

### Data Pipeline Summary

| Stage | Input | Output |
|---|---|---|
| Raw data | 100,000 books | 100,000 records |
| Format cleaning | Remove Non-Book entries | 99,941 records |
| Encoding filtering | Drop non-ASCII rows (~35%) | ~64,000 records |
| Word count filter | Keep descriptions ≥ 30 words | ~52,182 records |
| Language filter | langdetect + stop-word scoring | 52,182 records |
| Deduplication | title_clean + author_clean | 52,154 records |
| Genre filter | Drop missing genre | **47,161 final records** |

---

## Data Preprocessing

### Books
- **Format standardization**: Consolidated 202 bookformat variations → 8 macro-categories (Paperback, Hardcover, Ebook, etc.). Dropped 59 non-book entries (DVDs, Tarot cards).
- **Genre parsing**: Extracted, lowercased, deduped genre labels from pipe-separated multi-label strings. Applied GENRE_ALIAS mapping (e.g., `sci-fi` → `science fiction`).
- **Description cleaning**: Removed HTML tags, URLs, control characters; enforced ASCII-only; filtered to ≥ 30 words; validated with `langdetect` + English stop-word ratio ≥ 0.1.
- **Outlier handling**: Retained power-law distributions for ratings/popularity (used for Bayesian smoothing); no trimming.

### Reviews
- Crawled 87,051 reviews; removed 5,093 exact duplicates and 423 same-user/book duplicates.
- Translated non-English reviews; dropped failed translations → **75,801 clean English reviews**.
- Extracted features: `sentiment_polarity`, `sentiment_subjectivity`, `sentiment_label` (TextBlob); `word_count`, `is_long_review`.

### Interactions
- Simulated from review data using behavioral probability rules conditioned on ratings.
- Event types: **review** (explicit), **save**, **click**, **search**, **purchase**, **share** (implicit).
- Final dataset: **450,020 interactions** from 33,643 users across 6,001 books.

---

## Exploratory Data Analysis

Key findings that shaped the system design:

- **Ratings are tightly clustered** (mean 3.90, IQR 0.43) → raw rating alone insufficient for ranking; Bayesian smoothing applied.
- **Popularity follows a power-law distribution** (median totalratings = 170) → log-transform used; niche books still surfaced via content signals.
- **84.5% positive sentiment** across reviews → sentiment polarity used as cold-start quality proxy for books with ≤ 5 reviews (28% of catalog).
- **Genre co-occurrence** reveals strong overlaps: fantasy–paranormal (32%), fantasy–romance (30%) → multi-genre retrieval supported.
- **liked_ratio and avg_rating** are highly correlated (r = 0.886) → liked_ratio used as primary feature; avg_subjectivity removed (near-zero correlation).
- **Keyword long-tail distribution**: top 100 universal keywords cover most trope vocabulary; 54 cluster-mined tropes supplement coverage.

---

## NLP Pipeline

### Query-Side Processing

| Step | Method | Purpose |
|---|---|---|
| Text normalization | `normalize_query()` + `SYNONYM_MAP` | Unify synonyms (e.g., `sad` → `emotional`, `lgbt` → `lgbtq`) |
| Tokenization | spaCy `en_core_web_sm` | Tokenize, lemmatize, POS tag |
| Negation detection | Dependency tree scope traversal | Detect negated tokens (e.g., "without romance" → `not_romance`); apply −0.10 penalty per hit |
| Keyword extraction | Bigram-first greedy algorithm | Preserve compound phrases ("coming of age", "slow burn") |
| Layer 1 intent | `TROPE_KEYWORDS` + `TONE_KEYWORDS` dict matching | Activate tropes/tones from keyword triggers; deactivate if negated |
| Layer 2 intent | Zero-shot cosine similarity (threshold ≥ 0.45 tropes, ≥ 0.40 tones) | Handle abstract/sparse queries via semantic proximity |
| Query embedding | `all-MiniLM-L6-v2` + L2 normalization | Produce (1, 384) unit vector for similarity computation |

### Corpus-Side Processing

| Step | Method | Output |
|---|---|---|
| TF-IDF extraction | `max_features=6000`, `ngram_range=(1,2)` | Top-15 keywords per book |
| K-Means clustering | TruncatedSVD (100D) → K=60 clusters | 54 labeled `mined_trope` tags |
| Dual embedding | `all-MiniLM-L6-v2` | `desc_emb.npy` (1,384) + `review_emb.npy` (1,384) per book |
| Sentiment aggregation | TextBlob per review → book-level avg | `avg_polarity`, `avg_subjectivity`, `liked_ratio` |

### TROPE_LABELS Library
- **70+ curated labels** across 6 categories: relationship/romance, character arc, plot structure, world/setting, emotional/thematic, narrative device
- **54 mined labels** from unsupervised K-Means clustering (e.g., "tudor and british history", "manga romance", "pirates and nautical adventure")

---

## Recommender System

### Collaborative Composite Score

```
collab_score = 0.5 × bayesian_score
             + 0.3 × interaction_score
             + 0.2 × sentiment_score
```

| Sub-score | Weight | Source |
|---|---|---|
| `bayesian_score` | 0.5 | `books.rating` + `books.reviews` (Goodreads aggregate) |
| `interaction_score` | 0.3 | save, click, search, purchase, share (interactions.csv) |
| `sentiment_score` | 0.2 | `avg_polarity`, `1 - avg_subjectivity` (reviews) |

**Bayesian smoothing formula**: `(R × v + m × C) / (v + C)`  
where R = book avg rating, v = num ratings, m = global mean rating, C = global mean count.  
This prevents books with few ratings from dominating over well-established titles.

### User-Item Matrix (Collaborative Filtering)
- Active users (≥ 3 reviews): **3,938 users**, 39,647 interactions
- User–item matrix: shape (3938, 5090), **sparsity 99.8%**
- User similarity computed via cosine similarity; top-10 similar users used for CF blending

### Final Blend

```
final_score = 0.75 × content_score + 0.25 × collab_score
            − 0.10 × neg_similarity × n_neg_keywords
final_score = clip(final_score, 0, 1)
→ exclude already-read books → sort → top-N
```

---

## Evaluation & Results

### Metrics

| Metric | Definition | Weight in Composite |
|---|---|---|
| NDCG@5 | Normalized Discounted Cumulative Gain at top-5 | 0.35 |
| Trope-Precision@5 | Fraction of top-5 results matching query tropes | 0.25 |
| Catalog Coverage | Fraction of catalog appearing in ≥ 1 recommendation | 0.25 |
| Novelty | Fraction of long-tail books (bottom 20% popularity) recommended | 0.15 |

### Grid Search — Parameter Configurations

| Config | Fusion Mode | w_content | w_collab | trope_boost | Composite (VAL) |
|---|---|---|---|---|---|
| **D_content_heavy** ✅ | fixed | **0.75** | **0.25** | **+0.15** | **0.5586** |
| F_adaptive_content_max | adaptive | 0.75 | 0.25 | +0.15 | 0.5580 |
| C_less_collab_more_trope | fixed | 0.70 | 0.30 | +0.10 | 0.5548 |
| E_adaptive_balanced | adaptive | 0.70 | 0.30 | +0.10 | 0.5531 |
| A_baseline | fixed | 0.60 | 0.40 | +0.05 | 0.5334 |
| B_adaptive_fusion | adaptive | 0.60 | 0.40 | +0.05 | 0.5308 |

### Best Config vs Baseline (D_content_heavy vs A_baseline)

| Metric | Improvement |
|---|---|
| NDCG@5 | +0.0105 |
| Trope-Precision@5 | +0.0240 |
| Catalog Coverage | +0.0940 |
| Novelty | −0.0024 |
| **Composite** | **+0.0252** |

### Final Report — VALIDATE vs TEST

| Phase | NDCG@5 | Trope-Prec@5 | Coverage | Novelty | Composite |
|---|---|---|---|---|---|
| VALIDATE (batch 1, #1–#1000) | 0.5160 | 0.3354 | 0.7080 | 0.9692 | 0.5586 |
| TEST (batch 2, #1001–#2000) | 0.4992 | 0.3604 | 0.7020 | 0.9796 | 0.5601 |
| **Delta** | −0.0168 | +0.0250 | −0.0060 | +0.0104 | **+0.0015** |

> ✅ All deltas within the ±0.05 stability threshold — config generalizes well to unseen data.

---

## Prototype

The system is deployed as an interactive book search engine built with **Streamlit**, allowing users to input free-text natural language queries and receive real-time recommendations with content score, collaborative score, and final ranking score displayed per result.

**Demo queries tested:**
- `"i need a horror romance detective book that i couldn't put it down"` → Fatal Deception (0.6353), The Locked Room (0.6041), The Burning (0.5788)
- `"slow burn romance enemies to lovers"` → On Fire (0.8702), Wild Cat (0.8424), Chained by Night (0.8111)
- `"post-apocalyptic survival with political intrigue"` → Dead Inside (0.8171), This Book is Full of Spiders (0.7999), We Live Inside You (0.7845)
- `"a story where love is not returned"` → Eve's Daughters (0.5720), Great Tales of Terror (0.5490), Checkmate (0.5456)

---

## Project Structure

```
.
├── notebooks/
│   ├── 00_Data_Preprocessing.ipynb   # Data cleaning, feature engineering
│   ├── 01_Model.ipynb                # NLP pipeline + recommender system + grid search
│   └── 02_EDA.ipynb                  # Exploratory data analysis
├── data/
│   ├── 6000_bookssamples.csv         # Book metadata
│   ├── review_clean.csv              # Cleaned user reviews
│   └── interactions.csv              # Interaction events
├── embeddings/
│   ├── desc_emb.npy                  # Book description embeddings (N, 384)
│   ├── review_emb.npy                # Book review embeddings (N, 384)
│   └── query_emb.npy                 # Real user query embeddings
├── app/
│   └── streamlit_app.py              # Streamlit prototype
├── reports/
│   └── 03_BA_FINAL_REPORT.pdf        # Full project report
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/nlp-book-recommender.git
cd nlp-book-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### `requirements.txt`

```
pandas
numpy
scikit-learn
sentence-transformers
spacy
textblob
matplotlib
seaborn
streamlit
langdetect
```

---

## Usage

### Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### Use the recommender directly in Python

```python
from recommender import recommend, extract_intent_final

# Anonymous query
results = recommend(
    query="a dark fantasy with redemption arc, no romance",
    top_n=5
)
print(results[['title', 'genre_primary', 'final_score']])

# Personalized query (known user)
results = recommend(
    query="emotional story with found family",
    user_id=246,
    top_n=5
)
```

### Example output

```
Query: "slow burn romance enemies to lovers"
─────────────────────────────────────────────────────────────
Rank  Title                Content  Collab  Final   ★     #Ratings
1     on fire              0.8702   0.7540  0.8451  4.21  312
2     wild cat             0.8424   0.7210  0.8116  4.15  287
3     chained by night     0.8111   0.6980  0.7823  4.08  461
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| NLP | spaCy (`en_core_web_sm`), SentenceTransformers (`all-MiniLM-L6-v2`), TextBlob, langdetect |
| ML / Embeddings | scikit-learn (TF-IDF, KMeans, TruncatedSVD, MinMaxScaler, cosine similarity) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Prototype | Streamlit |

---

## Team

| Name | Student ID |
|---|---|
| Nguyễn Thị Minh Thư | 2311560045 |
| Lý Vân Ngọc | 2314560032 |
| Vũ Minh Trang | 2313560046 |
| Vũ Minh Thư | 2312560044 |
| Phạm Lương Duyên | 2313560009 |
| Hồ Hà Linh | 2312560025 |
| Lê Minh Anh | 2312560002 |

---

## References

- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
- Honnibal et al. (2020). *spaCy: Industrial-strength NLP in Python.*
- Gomez-Uribe & Hunt (2015). *The Netflix Recommender System.* ACM TMIS.
- Abdollahpouri et al. (2019). *The Unfairness of Popularity Bias in Recommendation.* arXiv.
- Manning, Raghavan & Schütze (2008). *Introduction to Information Retrieval.* Cambridge University Press.
