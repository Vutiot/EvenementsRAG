you are a UX/UI expert.

i want you to create a "parameter optimization" UI for my RAG project.

There should be two modes : First mode : Question menu with a chat for either asking a question either choosing a predefined question with a dropdown menu. If a predefined question was chosen, text is filled in the chat. Secund mode : Evaluation mode with no question interface 


After comes all parameters choices. The goal is to optimize one parameter when other are fixed.
use @rag-optimiser.jsx as a base template, but adapted to described parameters. also all UI has to be in light mode.

When validated you output :

if first mode : retrieval rank of chunk and doc, retrieval latency and generation metrics for the question, Top retrieved chunks.
if secund mode : retrieval, latency and generation metrics for the eval question dataset. Then a dropdown menu to select a question a get same overview as in first mode for this question.
