# financial-knowledge-clustering

This project provides a full pipeline that:
1. Accepts a list of financial article URLs (from a CSV like `example_input_sources.csv`),
2. Scrapes content from the articles and extracts a title, content, and summary for each,
3. Uses natural language processing to extract financial concepts and causal relationships,
4. Builds a knowledge graph in a Neo4j database, and
5. Applies clustering algorithms to group related financial knowledge.

## Features

- Web scraper to extract article content from provided URLs
- Transformer-based summarization and NLP processing
- Subject-Verb-Object pattern matcher for causal link extraction
- Neo4j integration for building and querying the graph
- Multiple clustering algorithms (KMeans, GMM, Agglomerative, etc.)
- Dimensionality reduction and visualization (t-SNE)

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fin_catch_pipeline_clean.py --input path/to/your_input.csv
```

The input CSV should have the following format:

```csv
source,URL
wiki,https://en.wikipedia.org/wiki/Currency
investopedia,https://www.investopedia.com/terms/f/financial_planning.asp
...
```

The script will output an intermediate CSV file named `Extracted_Financial_Articles.csv` with the following columns:
- `url`
- `title`
- `content`
- `summary`

From there, it will extract causal relationships and build a Neo4j graph, then cluster concepts based on text content + graph structure.

## Neo4j Setup

Make sure to set your Neo4j credentials using environment variables or secrets. For example:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
```

## Project Structure

```
fin_catch_pipeline_clean.py     # Main pipeline script
Extracted_Financial_Articles.csv # Intermediate output from URL scraping
README.md
```

## License

MIT License


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
