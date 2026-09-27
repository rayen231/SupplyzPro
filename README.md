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

RESULTS_PLACEHOLDER

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
