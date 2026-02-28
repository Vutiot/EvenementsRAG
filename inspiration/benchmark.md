my new goal is to create benchmarks to compare configs, and for that i will start by adding new parameters to benchmark. create a plan for it first. 2nd step will be to create a UI menu to test the rag system for a specific query and a specific config, but also help visualizing a benchmark and samples for a specific parameter with all other parameters fixed. you will have to check what was implemented and what wasn't in the project.


## "parameters" i want to benchmark

  dataset : wikipedia-scraping vs octank financial

  vector database : pg vector vs faiss vs qdrant

  chunk size

  chunk overlap

  Similarity metrics : cosine euclidean dot_product manhattan ANN (list and document different ANN methods )

  embeddings : bge-base vs miniLM-L12

  rag techniques : dense vs sparse vs hybrid ( date + keywords extraction thanks to llm, TF IDF,
  BM25) vs LazyGraphRAG

  reranker type :
  none cohere-v3 bge-reranker-v2 cross-encoder

  suggest any other parameters i could measure


## generation

top_k chunk extract

top_k documents extract

model for generation (free open router models)

prompt for generation (generate relevant prompt to test)

suggest any other parameters i could measure


## metrics i want in my benchmark interface :

  metrics for retrieval : article hit@k, chunk hit@k, MRR

  metrics for generation : ML metrcis (BLEU, Rouge, BERTScore, other semi-auto metrics)
  vanilla llmasjudge ( mistral small, which metrics ? ) and **ALL FOLLOWING ragas lib metrics**:
  faithfulness,answer_relevancy,context_precision,context_recall,context_entity_recall,an
  swer_similarity,answer_correctness,harmfulness,maliciousness,coherence,correctness,conc
  iseness

  metrics for latency : p99, p95, **suggest other**

## UI spec

As described in @UI.md **WITHOUT TAKING INTO ACCOUNT** parameters to optimize :
for benchmark results there should be two modes : a compute mode and a secund precomputed, cached mode where all results are stored from a database. If parameter combination is not precomputed, compute it and add to the database.



also suggest anything i could have forgotten when you read the project folder.