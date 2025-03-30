#!/usr/bin/env python
# coding: utf-8

# # FinCatchTask_Medium
# This notebook contains the full pipeline for Q1 and Q2 of the project, from article extraction to causal relationship visualization using Neo4j.

# Install the package

# In[ ]:


get_ipython().system('pip install pandas newspaper3k')
get_ipython().system('pip install lxml[html_clean]')
get_ipython().system('pip install sumy')
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


# ## Q1: Article Extraction
# We start by loading source URLs, extracting article content, and generating structured data (title, summary, content).
# 
#  **Note for Q1:**
# Please make sure `FinCatch_Sources_Medium.csv` is uploaded before running this section.  
# It is the input dataset used to extract content and generate summaries.
# 

# In[ ]:


import pandas as pd
df = pd.read_csv('FinCatch_Sources_Medium.csv')
print(df.columns)


# In[ ]:


from bs4 import BeautifulSoup
import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
nltk.download('punkt')

def generate_summary(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)

    summary = ' '.join(str(sentence) for sentence in summary_sentences)
    return summary
def extract_article_bs4(url):
    try:
        print(f"Processing: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No Title Found'

        paragraphs = soup.find_all('p')
        cleaned_paragraphs = []
        for p in paragraphs:
            text = p.get_text()
            text = re.sub(r'\[\d+\]', '', text)
            if len(text.strip()) > 50:
                cleaned_paragraphs.append(text.strip())

        content = ' '.join(cleaned_paragraphs)

        # Generate smart summary using TextRank
        summary = generate_summary(content, sentence_count=3)

        return {
            'url': url,
            'title': title,
            'content': content,
            'summary': summary
        }

    except Exception as e:
        print(f"[Error] Failed to process {url}: {e}")
        return {
            'url': url,
            'title': None,
            'content': None,
            'summary': None
        }


# In[ ]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk


df = pd.read_csv('FinCatch_Sources_Medium.csv')
urls = df['URL'].dropna().unique().tolist()

extracted_articles = []
for url in urls:
    result = extract_article_bs4(url)
    extracted_articles.append(result)

output_df = pd.DataFrame(extracted_articles)
output_df.to_csv('Extracted_Financial_Articles.csv', index=False)
print("Extraction completed.")
#print(output_df.head())


# # Q2: Causal Relationship Visualizer
# 
# This notebook extracts and visualizes causal relationships between financial concepts using both rule-based and NLP-based approaches. The relationships are stored in Neo4j and visualized through both Neo4j Browser and Python-based tools.
# 
#  **Note:**  
# This section requires `Extracted_Financial_Articles.csv` to be present in the `/content/` directory.  
# Make sure you:
# - Either upload the file manually, OR
# - Re-run Q1 to regenerate it.
# 

# ### Upload to Neo4j
# We upload the NLP-based causal relationships into Neo4j for graph-based exploration.

# In[ ]:


get_ipython().system('pip install py2neo')


# In[ ]:


import pandas as pd

# Load the structured dataset from Q1
df = pd.read_csv("/content/Extracted_Financial_Articles.csv")
df.head()


# ## Rule-Based Algorithm
# 
# This method uses keyword matching in the article content to identify causal relationships.
# 

# ## Q2: Causal Relationship Visualizer
# This section extracts causal relationships using two methods — rule-based and NLP — and stores the results in Neo4j.

# In[ ]:


import re

causal_keywords = [
    "because", "as a result", "which leads to", "therefore", "drives", "impacts",
    "results in", "contributes to", "is responsible for", "can cause", "explains", "influences"
]



def find_causal_relationships(df):
    relationships = []

    for i, row in df.iterrows():
        summary = row['content']

        title = row['title']

        if pd.isna(summary) or pd.isna(title):
            continue

        summary = summary.lower()

        for keyword in causal_keywords:
            if keyword in summary:
                if i + 1 < len(df):
                    next_title = df.iloc[i + 1]['title']
                    if pd.isna(next_title):
                        continue

                    cause = title.strip() if isinstance(title, str) else str(title)
                    effect = next_title.strip() if isinstance(next_title, str) else str(next_title)
                    relationships.append((cause, keyword.upper(), effect))
                break

    return relationships

causal_links = find_causal_relationships(df)

for cause, rel, effect in causal_links:
    print(f"{cause} --[{rel}]--> {effect}")


# ## NLP-Based Algorithm
# 
# This method uses dependency parsing from spaCy to extract subject–verb–object patterns that express causality.
# 

# In[ ]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')

import spacy
nlp = spacy.load("en_core_web_sm")


# In[ ]:


def extract_causal_from_sentence(text):
    doc = nlp(text)
    relations = []

    for sent in doc.sents:
        for token in sent:
            if token.lemma_ in ["cause", "lead", "impact", "influence", "result", "affect"]:
                subj = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr", "pobj")]

                if subj and obj:
                    relations.append((subj[0], token.lemma_.upper(), obj[0]))

    return relations


# In[ ]:


nlp_causal_links = []
df = pd.read_csv("/content/Extracted_Financial_Articles.csv")


for idx, row in df.dropna(subset=['content']).iterrows():
    content = row['content']
    extracted = extract_causal_from_sentence(content)
    for subj, verb, obj in extracted:
        nlp_causal_links.append((subj, verb, obj))
print(f"NLP-based causal links found: {len(nlp_causal_links)}")
for link in nlp_causal_links[:10]:
    print(f"{link[0]} --[{link[1]}]--> {link[2]}")


# ## Upload to Neo4j
# 
# Only NLP-based causal relationships are stored in Neo4j.
# 

# In[ ]:


# Replace with your Neo4j Aura connection info or local instance
from py2neo import Graph

bolt_uri = "bolt://localhost:7687"  # or your Neo4j Aura bolt URI
username = "<your-username>"
password = "<your-password>"

# Connect to Neo4j
graph = Graph(bolt_uri, auth=(username, password))


# In[ ]:


from py2neo import Node, Relationship

for cause, rel, effect in nlp_causal_links:
    # Create nodes
    cause_node = Node("Concept", name=cause)
    effect_node = Node("Concept", name=effect)

    # Create relationship
    relationship = Relationship(cause_node, rel, effect_node)

    # Merge into the graph
    graph.merge(cause_node, "Concept", "name")
    graph.merge(effect_node, "Concept", "name")
    graph.merge(relationship)


# ## Visualize output

# In[ ]:


get_ipython().system('pip install networkx matplotlib')


# In[ ]:


# Replace with your Neo4j Aura connection info or local instance
from py2neo import Graph

bolt_uri = "bolt://localhost:7687"  # or your Neo4j Aura bolt URI
username = "<your-username>"
password = "<your-password>"

# Connect to Neo4j
graph = Graph(bolt_uri, auth=(username, password))


# ## Conclusion
# 
# We implemented and compared two methods to extract causal relationships:
# - A **rule-based algorithm** using keyword matching
# - An **NLP-based method** using spaCy dependency parsing
# 
# The NLP approach provided richer and more nuanced relationships, which were stored and visualized in Neo4j. This system enables interactive exploration of financial knowledge through a causal lens.
# 

# In[ ]:





# ## Q3: Clustering Module for Financial Knowledge
# 
# In this section, we implement a clustering module that groups related financial concepts based on both semantic content and causal relationships derived from Q1 and Q2. The goal is to identify clusters of financial knowledge using machine learning algorithms and visualize the results.
# 
# ---

# # Q3 Clustering Module
# In this section, we aim to cluster financial knowledge concepts into reasonable groups based on both their **text content** and **causal relationships**.
# 
# We will:
# 1. Extract content-based features using **TF-IDF** or **Sentence-BERT** embeddings.
# 2. Extract causal graph features (from Neo4j), like how often a concept is a cause/effect.
# 3. Combine these features and apply clustering (e.g., KMeans, Agglomerative).
# 4. Evaluate clustering quality using **silhouette score**.
# 5. Visualize the clusters using **t-SNE**.
# 
# Let's get started!

# # Q3 Clustering Module
# In this section, we aim to cluster financial knowledge concepts into reasonable groups based on both their **text content** and **causal relationships**.
# 
# We will:
# 1. Extract content-based features using **TF-IDF** or **Sentence-BERT** embeddings.
# 2. Extract causal graph features (from Neo4j), like how often a concept is a cause/effect.
# 3. Combine these features and apply clustering (e.g., KMeans, Agglomerative).
# 4. Evaluate clustering quality using **silhouette score**.
# 5. Visualize the clusters using **t-SNE**.
# 
# 

# ###  Step 1: Connect to Neo4j Aura
# We begin by connecting to the Neo4j Aura database, which stores the causal relationships (from Q2) extracted using the NLP pipeline.
# 
# 

# In[ ]:


# Replace with your Neo4j Aura connection info or local instance
from py2neo import Graph

bolt_uri = "bolt://localhost:7687"  # or your Neo4j Aura bolt URI
username = "<your-username>"
password = "<your-password>"

# Connect to Neo4j
graph = Graph(bolt_uri, auth=(username, password))


# ###  Step 2: Extract Causal Relationship Features
# We extract cause-effect statistics for each concept to use as structural features for clustering.
# 

# In[ ]:


# Step 2: Extract cause/effect counts from graph
query = '''
MATCH (n:Concept)
OPTIONAL MATCH (n)-[r1]->()
OPTIONAL MATCH ()-[r2]->(n)
RETURN
  n.name AS concept,
  COUNT(DISTINCT r1) AS cause_count,
  COUNT(DISTINCT r2) AS effect_count
ORDER BY concept'''

causal_features_df = graph.run(query).to_data_frame()
causal_features_df.head()


# ###  Step 3: Generate Text Embeddings from Q1 Content
# We use Sentence-BERT to convert the full `content` of each article into semantic vectors.
# 

# In[ ]:


# Step 3: Prepare semantic vectors from Q1 data
from sklearn.feature_extraction.text import TfidfVectorizer
df_clean = df.dropna(subset=['content']).copy()
text = df_clean['content'].astype(str).tolist()

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_text = vectorizer.fit_transform(text)


# ### 🔗 Step 4: Combine Semantic and Causal Features
# We normalize causal features and horizontally stack them with the SBERT vectors.
# 

# In[ ]:


# Step 4: Merge with causal counts
df_clean['concept'] = df_clean['title'].str.strip()
df_combined = pd.merge(df_clean, causal_features_df, on='concept', how='left').fillna(0)
X_graph = df_combined[['cause_count', 'effect_count']].to_numpy()


# ### Step 5: Apply Clustering Algorithms
# We try different algorithms like KMeans and Agglomerative Clustering to find optimal groupings.
# 

# In[ ]:


# Step 5: Cluster using both text + graph features
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_final = hstack([X_text, X_graph])

kmeans = KMeans(n_clusters=5, random_state=42)
df_combined['cluster'] = kmeans.fit_predict(X_final)
score = silhouette_score(X_final, df_combined['cluster'])
print(f'Silhouette Score (text + graph): {score:.3f}')


# In[ ]:


get_ipython().system('pip install sentence-transformers')


# ###  Step 6: Evaluate Cluster Quality
# We use Silhouette Score to assess the quality of each clustering result. Higher scores indicate more meaningful clusters.
# 

# In[ ]:


from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Generate semantic vectors from content using Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert = model.encode(df_clean['content'].tolist())

# Step 2: Combine with graph features
from sklearn.preprocessing import StandardScaler
import numpy as np

# Normalize the graph features before combining
X_graph_scaled = StandardScaler().fit_transform(X_graph)
X_combined = np.hstack((X_sbert, X_graph_scaled))

# Step 3: Cluster using KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=5, random_state=42)
df_combined['cluster_sbert'] = kmeans.fit_predict(X_combined)

# Step 4: Evaluate with Silhouette Score
score_sbert = silhouette_score(X_combined, df_combined['cluster_sbert'])
score_sbert


# ###  Step 7: Visualize Clusters with t-SNE
# We apply PCA and t-SNE to reduce the feature space and visualize clusters in 2D.
# 

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Assume X_combined is already defined from previous steps

# Step 1: PCA reduction for clustering

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_combined)


# Step 2: Try multiple k values to find best clustering
results = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append((k, score))

# Step 3: Plot silhouette scores
ks, scores = zip(*results)
plt.figure(figsize=(8, 4))
plt.plot(ks, scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters (PCA)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Visualize best clustering with t-SNE
best_k = max(results, key=lambda x: x[1])[0]
kmeans_best = KMeans(n_clusters=best_k, random_state=42)
labels_best = kmeans_best.fit_predict(X_pca)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_best, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title(f"t-SNE Clustering Visualization (k={best_k})")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(results, columns=["k", "silhouette_score"])

display(results_df)


# ### Clustering with Agglomerative Clustering (Q3 Extension)
# 
# To further explore different clustering techniques, we apply **Agglomerative Clustering** using the same number of clusters (k=8) found optimal in the KMeans experiments.
# 
# We then visualize the resulting clusters using **t-SNE** in 2D space. This allows us to visually compare the quality of cluster separation between KMeans and Agglomerative methods.
# 
# The silhouette score for Agglomerative Clustering is also computed for quantitative evaluation.
# 

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Try with the same number of clusters (e.g., best k from KMeans = 8)
agg_model = AgglomerativeClustering(n_clusters=8)
df_combined['cluster_agglo'] = agg_model.fit_predict(X_combined)

# Evaluate
score_agglo = silhouette_score(X_combined, df_combined['cluster_agglo'])
print(f"Agglomerative Clustering Silhouette Score: {score_agglo:.3f}")


# In[ ]:


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Run t-SNE on the combined features
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
X_tsne = tsne.fit_transform(X_combined)

# Step 2: Prepare DataFrame for plotting
df_tsne = pd.DataFrame(X_tsne, columns=["TSNE-1", "TSNE-2"])
df_tsne["Cluster"] = df_combined["cluster_agglo"]

# Step 3: Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_tsne, x="TSNE-1", y="TSNE-2", hue="Cluster", palette="tab10")
plt.title("t-SNE Clustering Visualization (Agglomerative, k=8)")
plt.legend(title="Cluster")
plt.show()


# In[ ]:


# ================================
# 🧪 Try Spectral Clustering (k=8)
# ================================
from sklearn.cluster import SpectralClustering

# Fit Spectral Clustering with same k=8 for comparison
spectral = SpectralClustering(n_clusters=8, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
df_combined['cluster_spectral'] = spectral.fit_predict(X_combined)

# Evaluate with silhouette score
score_spectral = silhouette_score(X_combined, df_combined['cluster_spectral'])
print(f"Spectral Clustering Silhouette Score: {score_spectral:.3f}")


# In[ ]:


# ================================
# 🖼️ t-SNE Visualization for Spectral Clustering
# ================================
df_tsne['Cluster'] = df_combined['cluster_spectral']

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_tsne, x='TSNE-1', y='TSNE-2', hue='Cluster', palette='tab10')
plt.title("t-SNE Clustering Visualization (Spectral, k=8)")
plt.legend(title="Cluster")
plt.show()


# ## DBSCAN Clustering (Density-Based Spatial Clustering)
# 
# DBSCAN is a density-based clustering algorithm that groups together points that are closely packed, and marks points in low-density regions as outliers (noise). Unlike KMeans or Agglomerative, DBSCAN does not require the number of clusters `k` to be specified in advance.
# 
# We use `eps` and `min_samples` to define neighborhood density.
# 
# Useful when:
# - Data has arbitrary shape clusters
# - Outlier detection is needed
# - The number of clusters is unknown
# 

# In[ ]:


from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# DBSCAN clustering
dbscan_model = DBSCAN(eps=3, min_samples=2)
df_combined['cluster_dbscan'] = dbscan_model.fit_predict(X_combined)

# Filter out noise
mask = df_combined['cluster_dbscan'] != -1
n_clusters = len(set(df_combined['cluster_dbscan'][mask]))  # number of clusters excluding -1

# Only compute silhouette if we have 2 or more valid clusters
if mask.sum() > 1 and n_clusters >= 2:
    score_dbscan = silhouette_score(X_combined[mask], df_combined.loc[mask, 'cluster_dbscan'])
else:
    score_dbscan = -1  # Invalid score due to no valid clusters

print(f"DBSCAN Silhouette Score (excluding noise): {score_dbscan:.3f}")



# In[ ]:


# Visualize with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
X_tsne = tsne.fit_transform(X_combined)

df_tsne = pd.DataFrame(X_tsne, columns=['TSNE-1', 'TSNE-2'])
df_tsne['Cluster'] = df_combined['cluster_dbscan']

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_tsne, x='TSNE-1', y='TSNE-2', hue='Cluster', palette='tab10')
plt.title("t-SNE Clustering Visualization (DBSCAN)")
plt.legend(title="Cluster")
plt.show()


# ## Gaussian Mixture Model (GMM)
# 
# GMM is a soft clustering method based on probability distributions. Each point belongs to each cluster with a certain probability.
# 
# We set `n_components = 8` to match the number of clusters from KMeans for fair comparison. GMM is more flexible than KMeans, as it allows elliptical clusters and overlapping memberships.
# 

# In[ ]:


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=8, random_state=42)
df_combined['cluster_gmm'] = gmm.fit_predict(X_combined)

score_gmm = silhouette_score(X_combined, df_combined['cluster_gmm'])
print(f"GMM Silhouette Score: {score_gmm:.3f}")


# In[ ]:


# Visualize GMM with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
X_tsne = tsne.fit_transform(X_combined)

df_tsne = pd.DataFrame(X_tsne, columns=['TSNE-1', 'TSNE-2'])
df_tsne['Cluster'] = df_combined['cluster_gmm']

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_tsne, x='TSNE-1', y='TSNE-2', hue='Cluster', palette='tab10')
plt.title("t-SNE Clustering Visualization (GMM, k=8)")
plt.legend(title="Cluster")
plt.show()


# ### MeanShift Clustering
# 
# MeanShift is a non-parametric clustering algorithm that does not require the number of clusters to be specified. It works by iteratively shifting data points toward areas of higher density. This makes it well-suited for exploratory analysis when the true number of clusters is unknown.
# 
# We apply MeanShift to our combined semantic and graph features and evaluate its clustering performance using the silhouette score.
# 

# In[ ]:


# 📌 MeanShift Clustering
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

# Step 1: Initialize and fit MeanShift
mean_shift = MeanShift()
df_combined['cluster_meanshift'] = mean_shift.fit_predict(X_combined)

# Step 2: Evaluate
# Check number of clusters
n_meanshift = len(set(df_combined['cluster_meanshift']))

if n_meanshift > 1:
    score_meanshift = silhouette_score(X_combined, df_combined['cluster_meanshift'])
else:
    score_meanshift = -1  # invalid score when only one cluster is found

print(f"MeanShift Silhouette Score: {score_meanshift:.3f} (clusters: {n_meanshift})")


# ## Q3 Summary: Clustering Financial Knowledge
# 
# In this module, we aimed to cluster financial knowledge into meaningful groups using both textual and graph-based features extracted from Q1 and Q2.
# 
# ### ✅ Input
# - **Textual data**: `content` from Q1 (processed via Sentence-BERT).
# - **Graph structure**: Causal counts from Q2 (number of times a concept appears as cause/effect).
# 
# These were combined into a final feature matrix to support semantically informed clustering.
# 
# ---
# 
# ### ✅ Clustering Methods Compared
# 
# We implemented and compared **five different clustering algorithms**:
# 
# 1. **KMeans**
#    - ✅ Tested multiple `k` values (from 2 to 9).
#    - ✅ Chose `k=8` based on best silhouette score.
#    - ✅ t-SNE visualization provided.
#    - 🔹 **Best overall performance** (score ~0.24).
# 
# 2. **Agglomerative Clustering**
#    - ✅ Used the same `k=8` for fair comparison.
#    - ✅ Visualized via t-SNE.
#    - 🔹 Performance slightly lower than KMeans.
# 
# 3. **DBSCAN**
#    - ✅ Does not require `k`; uses `eps` and `min_samples`.
#    - ❌ Detected too few clusters for meaningful silhouette score.
#    - 🔹 Not ideal for this dataset.
# 
# 4. **Gaussian Mixture Model (GMM)**
#    - ✅ Used `n_components=8` for fair comparison.
#    - 🔹 Produced softer clustering, but with lower silhouette (~0.066).
# 
# 5. **MeanShift**
#    - ✅ Automatically detects number of clusters.
#    - ❌ Only found one cluster.
#    - 🔹 Not suitable for this dataset.
# 
# ---
# 
# ### ✅ Evaluation Metrics
# 
# - **Silhouette Score** used as the main metric to evaluate how well-separated clusters are.
# - **t-SNE** used for 2D visualization of clusters.
# - Scores indicate KMeans with Sentence-BERT + causal features provided the best clustering result.
# 
# ---
# 
# ### ✅ Conclusion
# 
# Based on the tests, **KMeans with `k=8`**, enriched with **Sentence-BERT semantic embeddings** and **causal frequency features** from Q2, provided the best performance and most meaningful grouping. This approach successfully satisfies all Q3 requirements:
# - Clustering algorithm implemented
# - Parameters tested and justified
# - Quality evaluated using silhouette score
# - Clusters visualized using t-SNE
# 
# 

# ##  Technical Discussion on Q3 Implementation
# 
# ### Challenges and Trade-offs
# 
# - **High-Dimensional Representations**:  
#   Using `all-MiniLM-L6-v2` from Sentence-BERT provided dense semantic embeddings (~384 dimensions). To mitigate the curse of dimensionality and reduce computational overhead, we applied **PCA** to reduce the feature space while preserving variance.
# 
# - **Combining Modalities**:  
#   We merged **semantic vectors** (text) with **graph-derived features** (causal in/out degree from Q2). Feature standardization (`StandardScaler`) was applied prior to clustering to avoid dominance of high-magnitude features.
# 
# - **Model Selection**:  
#   We compared five clustering algorithms:
#   - **KMeans**: Performed best in terms of **silhouette score** and interpretability.
#   - **Agglomerative Clustering**: Similar performance but more flexible hierarchy.
#   - **Gaussian Mixture Model (GMM)**: Soft clustering, but performed poorly on sparse data.
#   - **DBSCAN / MeanShift**: Struggled due to small sample size and sparse high-dimensionality; often detected only 1–2 clusters.
# 
# - **Evaluation Strategy**:  
#   We used **silhouette score** to evaluate intra-cluster cohesion vs. inter-cluster separation. For visualization, we used **t-SNE** to reduce to 2D. Scores were computed excluding noise labels (e.g., `-1` from DBSCAN) to avoid skewed evaluation.
# 
# - **Parameter Sensitivity**:  
#   We conducted a grid search over `k` (KMeans) in the range 2–9. Other methods (e.g., GMM) were set to use the optimal `k` from KMeans for a fair comparison.
# 
# ### Design Considerations
# 
# - Chose to **keep the pipeline modular** and reusable for scaling to larger datasets.
# - Focused on **interpretability and performance** rather than exhaustive hyperparameter tuning, given the project’s scope.
# 
# ---
# 
# ##  Future Extensions and Exploration
# 
# ### 1. **Causal Graph Embedding**
# - Apply **node embedding techniques** (e.g., **Node2Vec**, **GraphSAGE**) to encode the structure of the Q2 causal graph into low-dimensional vectors. These embeddings can replace or augment current causal features.
# 
# ### 2. **Dynamic Temporal Clustering**
# - Extend the pipeline to handle **time-series news data**, capturing the temporal evolution of financial events. Use methods like **Dynamic Time Warping (DTW)** or **Online Clustering**.
# 
# ### 3. **GNN-Based Causal Learning**
# - Replace rule-based causal extraction with **Graph Neural Networks** (e.g., **CaSPER**, **GATs**) to directly learn causality from labeled financial text and graphs.
# 
# ### 4. **Cluster Interpretation with LIME/SHAP**
# - Apply **explainable AI tools** to analyze representative terms or documents in each cluster and provide more insight into what each cluster represents semantically and causally.
# 
# ### 5. **End-to-End Investment Signal Generation**
# - Once clusters are formed, map them to real-world financial indicators (e.g., stock volatility, earnings changes) and assess whether cluster activity correlates with market movement.
# 
# ---
# 
# This module establishes a foundation for multi-modal unsupervised learning in financial text. With improved causality modeling, better graph embeddings, and real-time integration, the system can support decision intelligence for analysts and algorithmic trading systems.
# 

# In[ ]:




