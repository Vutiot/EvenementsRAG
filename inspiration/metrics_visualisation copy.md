

### Main changes

I want you to rebuild the bench and sweep visualisation tool.
major change here is to use the Apache EChart-for-react Library and change UX and graphs.

Completely delete the existing visualisation first with lateral config and every visualisations, start from a clean sheet.


Please replace with a Retrieval Evaluation section including following RAGAS Metrics :
*Context Precision
*Context Recall
*Context Relevance 


Also, append a Generation Evaluation section including RAGAS
*Faithfulness 
*Answer/Response Relevancy
*Factual Correctness

After this section add a valid option for Retrieval AND Generation evaluation to generate several times LLM metrics to stabalize results.  



### Sweep visualisation

Please just take inspiration from every @rag_sweep_bench features where you group by parameters. Instead of grouping by dropdown just display one graph after the other vertically, for each visualisation type.

Create a second interface in the same page by taking inspiration from @rag_sweep_all_metrics where you compare by bench runs.

### Bench visualisation

Just group all metric and here are visualisation and features you have to propose at least :

bar graphs with all metrics by group ( two retriever and generation quality sections one after the other aligned vertically). add possibility to filter to hide by clicking on each metric name.

for heatmaps make them colored but also propose Latex or markdown exports.


Whenever you click on a bench name in the runs page this should open the bench visualisation page. Same with sweeps.