please i want you to plan how to unify query tester,  runs, and sweeps in one interface and completely change the interface. This is a large modification. Do not implement but write a new roadmap_unification.md in the roadmap command style and add to refer to this new file too in local CLAUDE.md file.

here is the roadmap command /roadmap.

In next sections, will describe every features i want. If any missing suggest new ones. 


### Main purpose

## Menu fusion

Query, Benchmarks and Sweeps menus should be merged in a testing menu in first position.
Subtitle something like "Test Query, Benchmarks and Sweeps...(developp)"

This implies a switch button between Query, Benchmarks and Sweeps. 
Switching from Query to Benchmarks triggers the disappearing of the chat menu. I want you to propose several elegant solutions for this. please use /frontend to use correct frontend skills.



Benchmark and sweep interfaces are the same, the only difference is that for sweeps every single select parameter button are multi select button.

The logic behind sweep is that you test **EVERY** parameters combinations (cartesion product) when multiple choices are done.

## Simplified interface

Presets shouldn't be available anymore in Testing menu.

Only a button "New config" allows you to open config menu with presets in a first placed dropdown menu.
Default preset should be collection wiki_10k_qdrant_cs512_co128_minilm_l6_cosine with Top K=100 CrossEncoder TopKreranker=20 LLM Generation equals to Nemotron.

Center the request interface as presets are removed. please use /frontend to use correct frontend skills.

When executing Query Benchmark or sweep interface scrolls to the top.



### Additional features 


## For Query Mode

In query mode please use streaming to generate dynamically LLM generation output if chosen. Change place generation to last position and don't wait for it to finish before displaying chunks.


## For Benchmark runs

in benchmark runs section only keeps the table, no benchmark execution now in testing section. Please add evaluation dataset name after dataset column. 

please add a benchmark name in the runs section with default name equals to  collection name plus creation time.

also append **all tunable parameters measured in benchmarks** so everything except generation i guess. Also let fixed name column in first position and a horizontal scroll for exploring other columns.

The table should be for benchmarks AND sweeps. Follow description in @run_history_feature_spec.md for implementing this new feature conciliating benchmark and sweeps in same interface.




