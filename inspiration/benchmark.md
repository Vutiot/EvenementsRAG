my new goal is to create benchmarks to compare configs and i will add new parameters to benchmark. create a new doc for it first. 2nd step will be to create a UI menu to test the rag system for a specific query and a specific config, but also help visualizing a benchmark and samples for a specific parameter with all other parameters fixed. you will have to check what was implemented and what wasn't in the project. Do not design any UI for the moment but just check what is already in the project and what is not


  here are all "parameters" i want to benchmark

  dataset : wikipedia-scraping vs octank financial

  dataset-size : 1000 articles vs 10000

  vector database : pg vector vs faiss vs qdrant

  embeddings : bge-base vs miniLM-L12

  rag techniques : dense vs sparse vs hybrid ( date + keywords extraction thanks to llm, TF IDF,
  BM25) vs graph


  here are all metrics i want in my benchmark interface :

  metrics for retrieval : article hit@k, chunk hit@k, MRR

  metrics for generation : ML metrcis (BLEU, Rouge, BERTScore, other semi-auto metrics)
  vanilla llmasjudge ( mistral small, which metrics ? ) and **ALL FOLLOWING ragas lib metrics**:
  faithfulness,answer_relevancy,context_precision,context_recall,context_entity_recall,an
  swer_similarity,answer_correctness,harmfulness,maliciousness,coherence,correctness,conc
  iseness

  metrics for latency : p99, p95, **suggest other**


The UI should have specs as described in @UI.md :
for benchmark results there should be two modes : a first compute mode and a secund precomputed, cached mode where all results are stored from a 

when results have been computed :
benchmark


  suggest anything i could have forgotten when you read the project folder.