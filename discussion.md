# Project VITAL — RAG testing evidence

## Baseline

Codex ran three questions in fresh Streamlit AppTest sessions using the
original starter chatbot from commit `b749202` and local Ollama
(`qwen2.5:1.5b`, temperature 0). The starter sends only the question,
without retrieving clinic policies. These are retrospective baseline
runs using the pre-RAG code, recorded after implementation.

### How many hours before an appointment must I cancel?

> I'm sorry, but I can't answer this question as it doesn't provide enough context. Could you please provide more information or clarify the question?

### Can I request a same-day appointment?

> I'm sorry, but I can't assist with that.

### How are laboratory results communicated?

> Laboratory results are typically communicated in several ways, depending on the context and the nature of the results. Here are some common methods:
>
> 1. **Written Reports**: Laboratory results are often reported in written form, either as a report or as part of a larger report. This can be sent to the patient, the referring physician, or the patient's insurance company.
>
> 2. **Email or Electronic Health Records (EHR)**: Results can be sent electronically via email or integrated into electronic health records (EHR) systems. This is particularly useful for sharing results with multiple healthcare providers.
>
> 3. **Phone Calls**: Results can be communicated over the phone, especially if the results are urgent or require immediate action.
>
> 4. **In-Person Consultations**: In some cases, results may be discussed in person, either at the patient's request or as part of a follow-up appointment.
>
> 5. **Notifications**: Some laboratories may send notifications via text message or mobile app, especially for results that are not immediately actionable.
>
> 6. **Patient Education**: In some cases, the laboratory may provide educational materials or information about the results, especially if the results are complex or require further explanation.
>
> 7. **Follow-Up Appointments**: Results may be discussed during follow-up appointments, either in person or via telehealth.
>
> The method of communication is often determined by the urgency of the results, the preferences of the patient or healthcare provider, and the specific requirements of the laboratory and the healthcare system.

## What changed and why

- `rag.py` loads policy files in alphabetical order, ranks them with TF-IDF
  and cosine similarity, and labels each selected document with its filename.
  Sorting and filename tie-breaking make retrieval order reproducible.
- `app.py` adds the **Enable RAG** toggle and displays the retrieved files.
  With RAG OFF, the model receives only the question. With RAG ON, it also
  receives the labeled policy text and instructions to abstain when the
  evidence is insufficient. The toggle makes it possible to compare both
  modes with the same model and questions.

## RAG OFF and ON evidence

Codex ran each question in a fresh Streamlit AppTest session against local
Ollama using `qwen2.5:1.5b`, temperature 0, and a retrieval limit of two.
The comparison was repeated after the tokenizer change. The table shows
the final run, and the answer text was identical across runs for each mode.
The retrieved filenames and source-support judgments apply to RAG ON.

| Question | Expected source | Retrieved filenames | Response without RAG | Response with RAG | Supported by the source? |
| --- | --- | --- | --- | --- | --- |
| How many hours before an appointment must I cancel? | cancellations.txt | cancellations.txt, scheduling.txt | I'm sorry, but I can't answer this question as it doesn't provide enough context. Could you please provide more information or clarify the question? | You must cancel your appointment at least 18 hours before the scheduled time. | Yes. The policy says “at least 18 hours before the scheduled time.” |
| Can I request a same-day appointment? | scheduling.txt | scheduling.txt, portal.txt | I'm sorry, but I can't assist with that. | Yes, you can request a same-day appointment, but it must be available. | Yes. The policy says “Same-day appointments may be requested, subject to availability.” |
| How are laboratory results communicated? | lab_results.txt | lab_results.txt, portal.txt | Listed generic delivery methods including email, phone, and text. The identical full reply is in the Baseline section. | Laboratory results are communicated through the secure patient portal. | Yes. The policy says “Laboratory results are communicated through the secure patient portal.” |
| Does the clinic provide free transportation? | None | None | I'm sorry, but I can't answer this question because I don't have any specific information about a particular clinic. The answer would depend on the clinic's policies and the location. If you know the name of the clinic or the location, you can search online or contact them directly to find out if they provide free transportation. | I do not have enough information in the provided documents. | Appropriate abstention. The six policies contain no transportation information. |

## Observed failure and rerun

The transportation question exposed a retrieval failure. Plain TF-IDF
returned billing.txt (cosine similarity 0.270) and lab_results.txt (0.234).
Generic wording such as “clinic” produced lexical overlap with unrelated
policies. Neither policy answered the question. The model appropriately
abstained even when it received those irrelevant sources.

Codex switched TF-IDF to a tokenizer that filters common words and repeated
clinic boilerplate and normalizes policy terms such as “canceled” and
“cancellation” to “cancel.” After the change, the transportation question
retrieved zero files and the model still abstained. The three answerable
questions kept their intended policy ranked first. The cancellation policy
already ranked first in this checkout, with a score of 0.266 before the
change and 0.578 afterward.

## What the tests establish

`.venv-brew/bin/python -m pytest -q` reported 6 passed before the change
and 8 passed after it. The two added regression tests check that the
transportation question returns no matches and that “When can I cancel?”
ranks the cancellation policy first. The suite also checks the six-document
count, source labels, empty inputs, and result ordering and limits.
The automated suite exercises retrieval and context formatting. The four
live comparisons provide evidence about those specific model replies.
Other questions and policy wording could produce different results.

## Does RAG automatically make the chatbot trustworthy?

No. The lab policy gave Qwen the evidence it needed to answer specifically
about the secure patient portal. TF-IDF can still miss relevant sources or
retrieve unrelated files, and Qwen can add unsupported details even when
it receives the right policy. Verify each claim against the policy text.

## Does caching help?

Caching the loaded policies, fitted vectorizer, and document vectors could
reduce retrieval work across questions. The app would still vectorize each
question, rank the policies, and generate a response. Policy edits would
need to invalidate the cache. Caching was discussed as an extension.
