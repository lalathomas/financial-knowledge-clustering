
# financial-knowledge-clustering

This project applies natural language processing (NLP) and network analysis to uncover structured insights from unstructured financial text data.

It automates the extraction of financial concepts and causal relationships from article content, builds a causal knowledge graph using Neo4j, and applies clustering algorithms to identify related groups of financial information.

---

## 🔧 Features

- **CSV-Based Input**: Accepts any CSV file containing financial news data with a `content` column.
- **Causal Extraction**: Uses Subject-Verb-Object (SVO)-based NLP to extract cause-effect relations.
- **Neo4j Integration**: Uploads extracted nodes and edges into a Neo4j graph database.
- **Clustering**: Combines text features and graph structure to cluster related financial concepts using KMeans.
- **Modular & Reusable**: Clean architecture designed for easy integration with other data pipelines.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> Required libraries: `spacy`, `sklearn`, `sentence-transformers`, `neo4j`, `pandas`, `numpy`, etc.

---

### 2. Run the Pipeline

```python
from fin_catch_pipeline_clean import run_pipeline

run_pipeline(
    csv_path="your_articles.csv",
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)
```

---

## 📁 Input Format

Your CSV file must have at least a `content` column:

| title | content |
|-------|---------|
| ...   | "The increase in interest rates caused a drop in housing demand..." |

---

## 📊 Output

- **Neo4j Graph**:
  - Nodes: Financial concepts
  - Relationships: Causal verbs (e.g. "cause", "impact", "lead to")
- **Cluster Labels**:
  - Cluster assignment for each concept
  - Printed silhouette score for cluster quality

---

## 🔬 Methods Used

- Sentence-BERT for semantic embeddings
- TF-IDF for text vectorization (optional)
- KMeans clustering with silhouette scoring
- Neo4j Cypher queries for graph-based features

---

## 📌 TODOs

- Add CLI interface
- Add visualizations for graphs and clusters
- Support for real-time API ingestion

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 🤝 Contributions

Feel free to fork and open pull requests to add support for more NLP techniques, clustering models, or graph analytics.
