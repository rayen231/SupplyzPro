# Agent Failure Intelligence for SupplyzPro

A semantic debugger for AI agents operating an ERP. It reads agent sessions (messages, tool calls, tool results),
finds cases where the agent **finished without a technical error but still failed the user**, groups related cases,
ranks what developers should investigate first, and shows the exact events that support each finding.

> User: "Get the latest invoice for Ahmed Ben Ali from ABC."
> Agent searches by name, gets two Ahmed Ben Ali records (ABC Distribution, XYZ Trading), fetches XYZ's invoice,
> and replies "Here is the latest invoice for Ahmed Ben Ali from ABC Distribution."
> No exception, no timeout, every API call succeeded. The analyzer reports **F002 wrong entity**, with the user
> message, the ambiguous search result, the wrong call and the final answer as evidence.

## What it detects

| Code | Failure | Detected by |
|---|---|---|
| F001 | Unsupported success claim (tool says pending/failed, or no write happened) | Hard rule + LLM |
| F002 | Wrong customer, product or warehouse | Soft rule + LLM confirmation |
| F003 | Incomplete request presented as finished | LLM |
| F004 | Repeated question the user already answered | Soft rule + LLM |
| F005 | No-progress tool loop | Soft rule + LLM |
| F006 | Wrong interpretation of a correct tool result | LLM |
| F007 | Unsupported inference (predictions, rankings, promises) | LLM |

Legitimate behaviour is a first-class outcome, not a failure: refined searches, retries after a timeout,
genuine clarification questions, honest "pending approval" reports and self-corrections are labelled
`success`, `recovered` or `clarification_needed`.

## How it works

```
trace ─► rules (≈0.1 ms) ─► triage ─► LLM judge (Groq, structured output) ─► merge policy ─► evidence validation
                                                                                   │
                         ranked issues ◄─ priority ◄─ signature grouping + TF-IDF sub-clusters
```

- **Trace-only.** The analyzer never reads ERP state. ERP data is used only to generate traces and ground truth.
- **Rules** (`app/detection/rules.py`) catch contradictions between tool evidence and the final answer. Hard rules
  (status contradiction) cannot be overridden. Soft rules (entity mismatch, re-ask, loop) can be downgraded by the LLM.
- **LLM judge** (`app/reasoning/`) gets a compressed, ID-tagged trace and must return a Pydantic verdict with
  `evidence_event_ids`. IDs that do not exist are rejected. The judge does not see rule output, so one cached verdict
  serves both the LLM-only and the hybrid configurations.
- **Confidence tiers.** High: a rule fired and the LLM agrees. Medium: one source fired with valid evidence
  (LLM-only findings are capped here). Low: sources disagree or evidence was partly invalid. Low findings and
  overridden rule suspicions are listed separately in the dashboard.
- **Grouping.** Findings are grouped by signature (failure type + entity kind), then TF-IDF + agglomerative
  clustering finds sub-patterns inside each group. Sentence embeddings + HDBSCAN are the scale-up path.
- **Priority** = affected sessions × impact weight (high 5, medium 3, low 1) × mean confidence. Impact is high for
  writes to purchase orders, invoices and inventory, medium for wrong reads, low for re-asks and loops.
  It is a prioritization heuristic, not an objective truth.
- **MLflow** tracks the analyzer itself: one run per configuration with precision, recall, F1, false positive rate,
  evidence grounding, tokens, latency and cost; LangChain autolog traces each judge call.

## Dataset

87 synthetic sessions over a SupplyzPro-like ERP (`data/erp/`) with deliberate look-alikes: two *Ahmed Ben Ali*
customers at different companies, *Coca-Cola 1L* vs *Coca-Cola Zero 1L*, three warehouses.

| Split | Sessions | Purpose |
|---|---|---|
| tuned | 72 | Rules were written against these |
| heldout | 15 | Paraphrased wording and 5 French sessions the rules never saw |

Ground truth (`evaluation/ground_truth.json`) stores the expected outcome, failure type and the key evidence events
for each session. **Evidence grounding rate** = share of true-failure findings whose cited events include at least
one gold key event.

## Results

Measured on the 87 sessions, with Groq-hosted models: `openai/gpt-oss-120b` as the judge and `openai/gpt-oss-20b` as the cheap comparison. Prompt version v2.

| Configuration | Precision | Recall | F1 | False positive rate | Type macro-F1 | Held-out F1 | Cost / 1k sessions |
|---|---|---|---|---|---|---|---|
| Rules only | 1.00 | 0.64 | 0.78 | 0.00 | 0.55 | 0.67 | $0 |
| LLM only, 20B | 1.00 | 0.71 | 0.83 | 0.00 | 0.64 | 0.75 | $0.19 |
| Hybrid, 20B | 1.00 | 0.73 | 0.85 | 0.00 | 0.65 | 0.75 | $0.19 |
| LLM only, 120B | 1.00 | 0.91 | 0.95 | 0.00 | 0.90 | 0.95 | $0.40 |
| **Hybrid, 120B** | **1.00** | **0.93** | **0.96** | **0.00** | **0.90** | **0.95** | **$0.39** |

- **No legitimate session was flagged** in any configuration: refined searches, timeout retries, honest pending reports, clarifications and legitimate re-asks were all left alone.
- **Evidence grounding is 100%.** Every true-failure finding cites at least one gold key event.
- **Recovery recall** is 1.00 for the hybrid versus 0.78 for the LLM alone. The deterministic recovery signal fixes the cases where the LLM calls a refined search plain success.
- **Rules alone miss the semantic types by design.** They catch no incomplete requests, wrong interpretations or unsupported inferences, and no French or paraphrased success claims. The LLM adds that recall.
- **The small model is not enough on its own.** It excuses speculative statements and paraphrased success claims. Rules lift it slightly, but the 120B judge is the better choice at about $0.40 per thousand sessions.

**Runtime and cost per session**, for the hybrid with the 120B judge:

| Step | Value |
|---|---|
| Deterministic rules | about 0.1 ms |
| LLM judge latency | about 1.2 s per call |
| Tokens per LLM call | about 1,240 in and 350 out |
| LLM calls per session | at most 1 |
| Grouping and ranking | about 2 s for the whole batch, mostly library import, offline |

**What still fails** with the hybrid 120B configuration:
- **Loops ending in "not found":** the loop rule fires on 3 sessions, but the judge calls them success because the agent honestly reported "not found". Soft rules are overridable by design, so these appear in the dashboard's override list rather than as failures. Making the loop rule non-overridable would recover all three with no false positives on this data.
- **Wrong interpretation typed as a success claim:** 2 cases are labelled unsupported success instead of wrong interpretation. They are still counted as failures.
- **Request already contained the answer:** 1 repeated question where the answer was already in the first message. Neither layer catches it.

**Triage filters almost nothing on this dataset.** It sends 99% of sessions to the LLM, because nearly every synthetic session contains a write, an ambiguous search or a non-final status. On real traffic, with many plain lookups, the skip rate should be higher. We have not measured it.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env              # add GROQ_API_KEY; rules-only mode works without it
python -m scripts.generate_data   # ERP + traces + ground truth
python -m evaluation.evaluate     # rules, llm, hybrid -> SQLite + MLflow  (add --model openai/gpt-oss-20b to compare)
streamlit run dashboard/app.py    # developer dashboard
uvicorn app.api.main:app          # HTTP API: /issues, /sessions, POST /analyze
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m pytest -q tests
```

Docker: `docker compose up --build` starts the dashboard (8501), API (8000) and MLflow UI (5000).

## Limitations

- **Synthetic data, one author.** The same team wrote the traces, the held-out variants and the rules. The held-out
  split reduces that bias but does not remove it. Real SupplyzPro traces are the next validation step.
- **Small sample.** Per-type metrics rest on 5 to 13 examples and have wide error bars.
- **English-centric rules.** French success claims fall through to the LLM. Entity matching is language-agnostic.
- **Trace-only view.** A wrong answer that is consistent with every tool result the agent saw is invisible.
- **LLM variance.** Verdicts are cached per prompt version; re-running with a new prompt can change results.
- **Rate limits.** On Groq's free tier the judge runs at about two calls per minute, which is why results are
  precomputed and the dashboard reads them from SQLite.

## Project layout

```
app/detection    rules, triage          app/reasoning   prompt, schema, Groq judge
app/services     pipeline, merge, store app/grouping    signature + TF-IDF clustering
app/ranking      priority score         app/api         FastAPI
scripts/         ERP data + generator   evaluation/     ground truth + evaluate.py (MLflow)
dashboard/       Streamlit app          tests/          unit tests
```
