# Question Types Taxonomy for Historical Events RAG

## Overview

This document categorizes question types for evaluating our Historical Events RAG system across World War II events. Each RAG phase (Vanilla, Temporal, Hybrid, Graph) excels at different question types, and this taxonomy guides both development and evaluation.

**Purpose:**
- Define comprehensive question categories for WW2 historical events
- Map question types to expected RAG phase performance
- Guide test dataset creation for systematic evaluation
- Enable quantitative comparison across RAG approaches

---

## Question Type Categories

### 1. Factual Questions

Questions seeking specific, verifiable facts from historical records.

#### 1.1 Simple Fact Retrieval
**Definition**: Single-hop queries requiring one piece of information from documents.

**Examples:**
- "When did D-Day occur?"
- "Where was the Battle of Stalingrad fought?"
- "Who was the Supreme Commander of Allied Forces in Europe?"
- "What year did World War II begin?"
- "How many countries signed the Tripartite Pact?"
- "What was the codename for the Normandy invasion?"
- "Who was the Prime Minister of the United Kingdom during most of WW2?"
- "What ship was sunk at Pearl Harbor?"
- "When did Germany invade Poland?"
- "Where did the Yalta Conference take place?"

**Characteristics:**
- Direct answer in single document chunk
- Low ambiguity
- Factoid-style responses
- Requires simple information extraction

**Expected Best Phase**: Phase 1 (Vanilla RAG)

---

#### 1.2 Complex Fact Retrieval
**Definition**: Multi-faceted queries requiring synthesis of multiple facts.

**Examples:**
- "What were the three main Axis powers?"
- "List all the major conferences between Allied leaders during WW2"
- "What were the primary causes of the Battle of Britain?"
- "What countries were occupied by Germany by 1941?"
- "What were the key provisions of the Atlantic Charter?"
- "Which generals commanded forces during Operation Market Garden?"
- "What were the main strategic objectives of Operation Barbarossa?"
- "List the major naval battles in the Pacific Theater"
- "What technologies were developed during the Manhattan Project?"
- "What were the terms of Japan's surrender?"

**Characteristics:**
- Requires aggregating information from multiple chunks
- May need list comprehension
- Often involves "what were", "list all", "identify the"
- Medium complexity synthesis

**Expected Best Phase**: Phase 3 (Hybrid RAG) - keyword matching helps identify all relevant instances

---

### 2. Temporal Questions

Questions focused on time, sequence, duration, and chronological relationships.

#### 2.1 Chronological Ordering
**Definition**: Questions about sequence and temporal relationships between events.

**Examples:**
- "What happened before the attack on Pearl Harbor?"
- "What major events occurred after the fall of France?"
- "Which battle came first: Midway or Coral Sea?"
- "What events led up to D-Day?"
- "What happened between the invasion of Poland and the Battle of Britain?"
- "Did the Battle of Stalingrad occur before or after the Battle of Kursk?"
- "What was happening in Europe when Japan attacked Pearl Harbor?"
- "What events followed the atomic bombing of Hiroshima?"
- "Which conference happened first: Yalta or Potsdam?"
- "What battles occurred in 1944?"

**Characteristics:**
- Requires temporal reasoning
- Keywords: "before", "after", "between", "first", "when"
- Needs chronological context
- Date comparison critical

**Expected Best Phase**: Phase 2 (Temporal RAG) - date filters and chronological sorting

---

#### 2.2 Duration and Timing
**Definition**: Questions about how long events lasted or when they peaked.

**Examples:**
- "How long did the Battle of Stalingrad last?"
- "What was the duration of the Siege of Leningrad?"
- "How many years did World War II span?"
- "When did the Battle of the Bulge reach its peak?"
- "How long did the Blitz last?"
- "What month did Germany surrender?"
- "How many days did the Dunkirk evacuation take?"
- "When was the turning point of the war in Europe?"
- "How long did Japan occupy the Philippines?"
- "What was the length of the Battle of Britain?"

**Characteristics:**
- Focus on temporal extent
- Keywords: "how long", "duration", "when did", "peak"
- Requires date arithmetic
- Need precise temporal metadata

**Expected Best Phase**: Phase 2 (Temporal RAG) - precise date extraction and calculation

---

#### 2.3 Causal Chains
**Definition**: Questions about cause-effect relationships and temporal dependencies.

**Examples:**
- "What events led to the United States entering World War II?"
- "What were the consequences of the Battle of Midway?"
- "How did the Treaty of Versailles contribute to World War II?"
- "What resulted from the fall of France in 1940?"
- "What caused the German offensive at the Battle of the Bulge?"
- "What were the outcomes of the Tehran Conference?"
- "How did Operation Barbarossa affect the Eastern Front?"
- "What led to Italy's surrender in 1943?"
- "What were the immediate effects of the atomic bombings?"
- "How did the Battle of Britain influence Germany's war strategy?"

**Characteristics:**
- Requires understanding causality
- Keywords: "led to", "caused", "resulted in", "consequences"
- Multi-hop reasoning needed
- Temporal + semantic connections

**Expected Best Phase**: Phase 4 (Graph RAG) - causal relationship edges in knowledge graph

---

### 3. Comparative Questions

Questions requiring comparison between two or more events, entities, or concepts.

#### 3.1 Event Comparison
**Definition**: Direct comparison of characteristics between events.

**Examples:**
- "How did the Battle of Stalingrad differ from the Battle of Kursk?"
- "What were the similarities between D-Day and the Sicily invasion?"
- "Compare casualties at Iwo Jima and Okinawa"
- "How did the Pacific Theater differ from the European Theater?"
- "What were the key differences between the Yalta and Potsdam conferences?"
- "Compare German and Allied tank technology in 1944"
- "How did British and American bombing strategies differ?"
- "Compare the fall of France with the fall of Poland"
- "What were the similarities between Operation Torch and Operation Husky?"
- "How did the Holocaust differ from other war atrocities?"

**Characteristics:**
- Requires retrieval of information about multiple subjects
- Keywords: "compare", "differ", "similarities", "versus"
- Needs parallel fact extraction
- Synthesis of contrasting information

**Expected Best Phase**: Phase 3 (Hybrid RAG) - retrieves all relevant mentions for comparison

---

#### 3.2 Superlative Questions
**Definition**: Questions seeking extremes or rankings.

**Examples:**
- "What was the deadliest battle of World War II?"
- "Which battle had the most casualties?"
- "What was the longest military campaign?"
- "Which air raid caused the most damage?"
- "Who was the highest-ranking German officer captured?"
- "What was the largest amphibious invasion?"
- "Which country suffered the most civilian casualties?"
- "What was the most decisive naval battle in the Pacific?"
- "Which bombing raid was the most devastating?"
- "What was the fastest Allied advance in Europe?"

**Characteristics:**
- Requires aggregation and ranking
- Keywords: "most", "largest", "highest", "deadliest", "longest"
- Needs comparison across all relevant entities
- Quantitative analysis

**Expected Best Phase**: Phase 3 (Hybrid RAG) - comprehensive retrieval + numerical comparison

---

### 4. Entity-Centric Questions

Questions focused on specific people, places, organizations, or military units.

#### 4.1 Person-Focused Questions
**Definition**: Questions about roles, actions, and characteristics of individuals.

**Examples:**
- "What role did Winston Churchill play in World War II?"
- "Who was Erwin Rommel?"
- "What battles did George Patton command?"
- "What happened to Adolf Hitler at the end of the war?"
- "Who were the key figures at the Tehran Conference?"
- "What was Dwight Eisenhower's strategy for D-Day?"
- "How did Charles de Gaulle contribute to the liberation of France?"
- "What was Douglas MacArthur's role in the Pacific?"
- "Who replaced Neville Chamberlain as British Prime Minister?"
- "What military positions did Bernard Montgomery hold?"

**Characteristics:**
- Named entity central to query
- Keywords: person names, "who", "what role"
- Requires entity linking
- May need biographical context

**Expected Best Phase**: Phase 3 (Hybrid RAG) - entity extraction and boosting

---

#### 4.2 Location-Focused Questions
**Definition**: Questions about events at specific geographic locations.

**Examples:**
- "What battles occurred in North Africa?"
- "What happened in Stalingrad during 1942-1943?"
- "Which cities were bombed during the Blitz?"
- "What military operations took place in the Pacific islands?"
- "What was the significance of the Ardennes Forest?"
- "What events occurred at Normandy beaches?"
- "Which concentration camps were in Poland?"
- "What battles were fought in Italy?"
- "What happened at Pearl Harbor on December 7, 1941?"
- "Which countries were neutral during World War II?"

**Characteristics:**
- Geographic location is query focus
- Keywords: place names, "where", "in [location]"
- Spatial reasoning helpful
- Location-event associations

**Expected Best Phase**: Phase 3 (Hybrid RAG) - location entity matching

---

#### 4.3 Organization-Focused Questions
**Definition**: Questions about military units, alliances, and institutional entities.

**Examples:**
- "What was the Wehrmacht?"
- "Which countries formed the Axis alliance?"
- "What was the role of the Office of Strategic Services (OSS)?"
- "What operations did the 101st Airborne Division participate in?"
- "What were the responsibilities of the Allied Control Council?"
- "What was the Luftwaffe's strategy during the Battle of Britain?"
- "Which nations were part of the Allied Powers?"
- "What was the SS (Schutzstaffel)?"
- "What role did the Royal Navy play in the Atlantic?"
- "What was Bletchley Park's function?"

**Characteristics:**
- Organizational entity focus
- Keywords: organization names, alliances, units
- May involve hierarchical relationships
- Institutional knowledge

**Expected Best Phase**: Phase 3 (Hybrid RAG) - entity recognition and matching

---

### 5. Relationship Questions

Questions about connections, influences, and interactions between entities and events.

#### 5.1 Influence and Impact
**Definition**: Questions about how events or entities affected others.

**Examples:**
- "How did the Battle of Midway influence the Pacific War?"
- "What impact did the atomic bomb have on Japan's surrender?"
- "How did the fall of France affect British strategy?"
- "What was the influence of the Soviet Union on the Eastern Front?"
- "How did radar technology impact the Battle of Britain?"
- "What effect did the U-boat campaign have on Allied supply lines?"
- "How did the Enigma code breaking influence the war outcome?"
- "What was the impact of American entry into the war?"
- "How did the Battle of Stalingrad affect German morale?"
- "What influence did air superiority have on D-Day success?"

**Characteristics:**
- Focuses on effects and influences
- Keywords: "influence", "impact", "effect", "affect"
- Requires understanding relationships
- Often multi-hop reasoning

**Expected Best Phase**: Phase 4 (Graph RAG) - relationship edges capture influence

---

#### 5.2 Network and Alliance Questions
**Definition**: Questions about collaborations, alliances, and networked relationships.

**Examples:**
- "Who were the Big Three Allied leaders?"
- "What alliances existed during World War II?"
- "Which generals worked together during Operation Overlord?"
- "What was the relationship between Churchill and Roosevelt?"
- "Which countries collaborated with Nazi Germany?"
- "How did the Allies coordinate their military strategies?"
- "What was the relationship between Hitler and Mussolini?"
- "Which resistance movements cooperated with Allied forces?"
- "How did the Soviet Union and Western Allies coordinate?"
- "What partnerships existed in the Manhattan Project?"

**Characteristics:**
- Multiple entities interconnected
- Keywords: "alliance", "collaboration", "partnership", "relationship"
- Network structure important
- Social/political connections

**Expected Best Phase**: Phase 4 (Graph RAG) - network structure in graph

---

#### 5.3 Multi-Hop Reasoning Questions
**Definition**: Questions requiring connecting information across multiple documents or facts.

**Examples:**
- "How did the invasion of Poland lead to the Battle of Britain?"
- "What chain of events connected the Treaty of Versailles to Hitler's rise?"
- "Trace the path from Normandy to the fall of Berlin"
- "How did early Allied losses lead to eventual victory?"
- "What connections existed between the European and Pacific theaters?"
- "How did espionage activities influence major battles?"
- "What events connected the fall of France to the Battle of El Alamein?"
- "How did technological developments influence battle outcomes?"
- "What links existed between the Holocaust and military operations?"
- "How did economic factors influence military decisions?"

**Characteristics:**
- Requires traversing multiple facts
- Keywords: "trace", "connect", "chain", "path"
- Complex reasoning paths
- Multiple inference steps

**Expected Best Phase**: Phase 4 (Graph RAG) - path finding in knowledge graph

---

### 6. Analytical Questions

Questions requiring synthesis, interpretation, or higher-level analysis.

#### 6.1 Synthesis Questions
**Definition**: Questions requiring summarization or aggregation of multiple concepts.

**Examples:**
- "Summarize the key events of 1944"
- "What were the main turning points of World War II?"
- "Describe the overall strategy of the Allied invasion of Europe"
- "What were the major developments in the Pacific Theater in 1945?"
- "Summarize the key decisions made at the Yalta Conference"
- "What were the primary factors in Allied victory?"
- "Describe the evolution of air warfare during WW2"
- "What were the main phases of the Battle of the Atlantic?"
- "Summarize the Holocaust and its scale"
- "What were the key technological innovations of the war?"

**Characteristics:**
- Broad scope
- Keywords: "summarize", "describe", "main", "key", "overall"
- Requires aggregation
- Narrative construction needed

**Expected Best Phase**: Phase 4 (Graph RAG) - can identify key nodes and patterns

---

#### 6.2 Interpretive Questions
**Definition**: Questions requiring inference, interpretation, or explanation beyond facts.

**Examples:**
- "Why was D-Day considered a turning point?"
- "What made the Battle of Stalingrad so significant?"
- "Why did Germany fail to invade Britain?"
- "What strategic mistakes did the Axis powers make?"
- "Why was the atomic bomb used on Japan?"
- "What factors led to the success of Operation Overlord?"
- "Why did the Blitzkrieg strategy succeed in France but fail in Russia?"
- "What was the strategic importance of North Africa?"
- "Why did Italy switch sides in 1943?"
- "What role did intelligence play in Allied victory?"

**Characteristics:**
- Requires reasoning beyond stated facts
- Keywords: "why", "significance", "importance", "explain"
- Interpretive analysis needed
- Context-dependent answers

**Expected Best Phase**: Phase 4 (Graph RAG) - contextual understanding from graph structure

---

#### 6.3 Counterfactual Questions
**Definition**: Hypothetical questions about alternative scenarios.

**Examples:**
- "What if Germany had won the Battle of Britain?"
- "How might the war have changed if the US didn't enter?"
- "What if Operation Barbarossa had succeeded?"
- "What would have happened if D-Day had failed?"
- "What if Japan hadn't attacked Pearl Harbor?"
- "How would the war differ without the atomic bomb?"
- "What if the Enigma code was never broken?"
- "What if Hitler had not invaded the Soviet Union?"
- "What might have happened if Churchill lost the 1940 election?"
- "What if the US had developed the atomic bomb later?"

**Characteristics:**
- Hypothetical/speculative
- Keywords: "what if", "would have", "might have"
- Requires understanding of causal chains
- No definitive answers possible

**Expected Best Phase**: Phase 4 (Graph RAG) - understanding causal relationships to reason about alternatives
**Note**: These questions may be beyond current RAG capabilities and serve as stretch goals

---

## RAG Phase Performance Mapping

### Phase 1: Classical Vanilla RAG

**Strengths:**
- Simple fact retrieval (1.1)
- Person-focused questions (4.1) - if well-described in single chunks
- Location-focused questions (4.2) - basic queries
- Straightforward entity queries

**Weaknesses:**
- Temporal reasoning
- Multi-hop questions
- Comparative questions requiring multiple retrievals
- Causal chains

**Expected Performance:**
- Factual questions: 80-90% accuracy
- Entity-centric questions: 70-80% accuracy
- Temporal questions: 40-50% accuracy
- Relationship questions: 30-40% accuracy

---

### Phase 2: Temporal RAG

**Strengths:**
- Chronological ordering (2.1) - major improvement
- Duration and timing (2.2) - significant improvement
- Time-bounded factual queries
- Sequential event questions

**Weaknesses:**
- Complex causal reasoning
- Multi-hop without temporal component
- Purely semantic comparisons
- Network relationships

**Expected Performance:**
- Temporal questions: 75-85% accuracy (30%+ improvement over Phase 1)
- Chronological questions: 80-90% accuracy
- Factual questions: 80-90% (maintains Phase 1 performance)
- Causal chains: 50-60% (some improvement on temporal causality)

---

### Phase 3: Hybrid RAG

**Strengths:**
- Complex fact retrieval (1.2) - keyword matching helps
- Event comparison (3.1) - retrieves all relevant mentions
- Superlative questions (3.2) - comprehensive retrieval
- All entity-centric questions (4.1, 4.2, 4.3) - entity matching
- Precision through reranking

**Weaknesses:**
- Deep causal reasoning
- Multi-hop inference
- Complex network questions
- Path-based reasoning

**Expected Performance:**
- Entity-centric questions: 85-95% accuracy (15-20% improvement)
- Comparative questions: 80-90% accuracy
- Complex factual: 85-90% accuracy
- Temporal questions: 75-85% (maintains Phase 2 gains)
- Multi-hop: 45-55% (slight improvement)

---

### Phase 4: Graph RAG

**Strengths:**
- Causal chains (2.3) - relationship edges
- Influence and impact (5.1) - explicit relationships
- Network questions (5.2) - graph structure
- Multi-hop reasoning (5.3) - path finding
- Synthesis questions (6.1) - identify key patterns
- Interpretive questions (6.2) - contextual understanding

**Weaknesses:**
- May have slightly lower precision on simple queries due to complexity
- Depends on graph construction quality
- Higher computational cost

**Expected Performance:**
- Causal chain questions: 80-90% accuracy (50%+ improvement over Phase 1)
- Multi-hop questions: 75-85% accuracy (45%+ improvement)
- Relationship questions: 80-90% accuracy (50%+ improvement)
- Network questions: 85-95% accuracy
- Maintains strong performance on all previous question types: 80%+

---

## Evaluation Question Set Design

### Distribution Across Categories

Recommended test set composition (175 questions total):

| Category | Simple | Medium | Hard | Total | % of Dataset |
|----------|--------|--------|------|-------|-------------|
| Factual (1.1, 1.2) | 10 | 10 | 5 | 25 | 14% |
| Temporal (2.1, 2.2, 2.3) | 15 | 15 | 10 | 40 | 23% |
| Comparative (3.1, 3.2) | 8 | 8 | 4 | 20 | 11% |
| Entity-Centric (4.1, 4.2, 4.3) | 12 | 12 | 6 | 30 | 17% |
| Relationship (5.1, 5.2, 5.3) | 12 | 15 | 8 | 35 | 20% |
| Analytical (6.1, 6.2) | 10 | 10 | 5 | 25 | 14% |
| **Total** | **67** | **70** | **38** | **175** | **100%** |

### Complexity Levels

**Simple (67 questions):**
- Single-hop reasoning
- Clear, unambiguous answers
- Information typically in one document
- Example: "When did D-Day occur?"

**Medium (70 questions):**
- 2-3 hop reasoning
- May require synthesis of 2-3 documents
- Some ambiguity or context needed
- Example: "What were the main causes of the Battle of Britain?"

**Hard (38 questions):**
- Multi-hop reasoning (3+ hops)
- Requires synthesis across many documents
- Complex causal or temporal reasoning
- Example: "How did the Treaty of Versailles contribute to the events leading to D-Day?"

### Question Template Examples

**Factual Template:**
```
- "When did [EVENT] occur?"
- "Where was [EVENT] located?"
- "Who was [ROLE] during [EVENT]?"
- "What were the [ATTRIBUTE] of [EVENT]?"
```

**Temporal Template:**
```
- "What happened before [EVENT]?"
- "What events occurred between [DATE1] and [DATE2]?"
- "How long did [EVENT] last?"
- "What caused [EVENT]?"
```

**Comparative Template:**
```
- "How did [EVENT1] differ from [EVENT2]?"
- "What were similarities between [EVENT1] and [EVENT2]?"
- "Which [EVENT] had more [ATTRIBUTE]?"
```

**Entity Template:**
```
- "What role did [PERSON] play in [EVENT]?"
- "What events occurred in [LOCATION]?"
- "What was [ORGANIZATION]'s involvement in [EVENT]?"
```

**Relationship Template:**
```
- "How did [EVENT1] influence [EVENT2]?"
- "What connections existed between [ENTITY1] and [ENTITY2]?"
- "Trace the path from [EVENT1] to [EVENT2]"
```

**Analytical Template:**
```
- "Summarize [TOPIC/PERIOD]"
- "What were the key [ASPECT] of [EVENT]?"
- "Why was [EVENT] significant?"
```

### Ground Truth Annotation

Each question in the evaluation set should include:

```json
{
  "id": "q042",
  "question": "What events led to the D-Day invasion?",
  "category": "temporal",
  "subcategory": "causal_chain",
  "complexity": "hard",
  "expected_best_phase": "phase4_graph",
  "ground_truth": {
    "answer": "The D-Day invasion was the result of multiple factors including Allied conferences (Tehran Conference in 1943 where it was planned), strategic bombing campaigns to weaken German defenses, Operation Fortitude (deception operations), and the need to open a Western Front to relieve Soviet pressure in the East.",
    "relevant_documents": [
      "World War II",
      "Operation Overlord",
      "Tehran Conference",
      "Operation Fortitude",
      "D-Day"
    ],
    "relevant_events": [
      "Tehran Conference",
      "Operation Fortitude",
      "Strategic bombing of Germany",
      "Soviet victories in the East"
    ],
    "entities": {
      "people": ["Eisenhower", "Churchill", "Roosevelt", "Stalin"],
      "locations": ["Normandy", "Tehran", "Britain"],
      "organizations": ["Allied Forces", "Wehrmacht"]
    },
    "dates": ["1943-11-28", "1944-06-06"],
    "reasoning_hops": 3,
    "evaluation_notes": "Requires understanding temporal sequence, causal relationships, and multiple planning events. Graph RAG should excel by following causal edges."
  }
}
```

---

## Metrics by Question Type

### Retrieval Metrics
- **Recall@K (K=5,10,20)**: Did we retrieve relevant documents?
- **MRR (Mean Reciprocal Rank)**: Position of first relevant document
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality

### Generation Metrics
- **Exact Match**: For factoid questions (dates, names)
- **ROUGE-L**: Lexical overlap with reference answer
- **BERTScore**: Semantic similarity with reference
- **Faithfulness (RAGAS)**: Answer grounded in retrieved docs
- **Answer Relevance (RAGAS)**: Answer addresses the question

### Specialized Metrics

**For Temporal Questions (Category 2):**
- Date extraction accuracy
- Chronological ordering correctness (% of correct sequences)
- Temporal span accuracy (correct duration calculations)

**For Comparative Questions (Category 3):**
- Coverage completeness (both entities covered in answer)
- Contrast clarity (differences explicitly stated)
- Factual accuracy for both subjects

**For Relationship Questions (Category 5):**
- Path correctness (for multi-hop questions)
- Relationship accuracy (correct edge types identified)
- Completeness of causal chain

---

## Usage Guidelines

### For Dataset Creation
1. Start with templates, fill with WW2 entities/events
2. Ensure even distribution across complexity levels
3. Validate each question has clear ground truth in Wikipedia sources
4. Include diverse time periods (1939-1945)
5. Balance between theaters (European, Pacific, etc.)

### For Evaluation
1. Run each RAG phase on the full test set
2. Compute metrics by category and complexity
3. Identify strengths/weaknesses of each phase
4. Validate hypothesis: later phases improve on targeted question types

### For Error Analysis
1. Manually review failed questions
2. Categorize failure modes (retrieval failure, generation error, etc.)
3. Identify patterns in failures by question type
4. Use insights to improve next phase

---

## Example Questions by Phase Performance

### Best for Phase 1 (Vanilla RAG)
1. "When did World War II begin?" (Simple factual)
2. "Who was the leader of Nazi Germany?" (Entity factual)
3. "Where was Pearl Harbor located?" (Location factual)
4. "What was the Blitzkrieg?" (Definition)
5. "Who commanded the Allied forces on D-Day?" (Person factual)

### Best for Phase 2 (Temporal RAG)
1. "What happened in 1944?" (Temporal range)
2. "What events occurred between Pearl Harbor and D-Day?" (Temporal range)
3. "Did the Battle of Midway occur before or after Guadalcanal?" (Chronological)
4. "How long did the Battle of Britain last?" (Duration)
5. "What were the events leading to the fall of France?" (Temporal causal)

### Best for Phase 3 (Hybrid RAG)
1. "List all the major conferences between Allied leaders" (Complex retrieval)
2. "What battles involved the 101st Airborne Division?" (Entity-based)
3. "Compare the Battles of Stalingrad and Kursk" (Comparison)
4. "Which battle had the most casualties?" (Superlative)
5. "What cities were bombed during the Blitz?" (Entity + keyword)

### Best for Phase 4 (Graph RAG)
1. "How did the fall of France lead to the Battle of Britain?" (Causal chain)
2. "What was the relationship between Churchill and Roosevelt?" (Relationship)
3. "Trace the Allied advance from Normandy to Berlin" (Path finding)
4. "How did the Battle of Midway influence the Pacific War?" (Influence)
5. "What connections existed between the Manhattan Project and Japan's surrender?" (Multi-hop)

---

## Conclusion

This taxonomy provides a comprehensive framework for:
1. **Developing** RAG systems with clear targets for each phase
2. **Evaluating** performance across diverse question types
3. **Comparing** different RAG approaches quantitatively
4. **Identifying** strengths and weaknesses of each implementation

As we build each RAG phase, we should continuously validate against this taxonomy and refine our evaluation dataset to ensure comprehensive coverage of historical event question types.

**Next Steps:**
1. Generate full 175-question test set using these templates
2. Create ground truth annotations for each question
3. Establish baseline with Phase 1 (Vanilla RAG)
4. Track improvements as we progress through Phase 2, 3, and 4
