# SQL Agent RL Environment

A multi-turn RL environment for LLM post-training, in which an agent answers
analytical business questions over a real relational database by inspecting the
schema, writing SQL, reading results, recovering from errors, and submitting a
final answer.

The interesting part of this project is **the environment and the reward
function**, not the SQL agent. Correctness dominates the reward, but the rubric
also shapes tool-use behaviour — inspect before querying, recover from errors,
don't spam calls — and every shaping term is designed and tested against a
specific reward-hacking mode.

```
                                    ┌──────────────────────────┐
                                    │  Chinook (real SQLite)   │
                                    └────────────┬─────────────┘
                                                 │
                   ┌─────────────────────────────┴──────────────────────────┐
                   │                                                        │
        ┌──────────▼───────────┐                              ┌─────────────▼────────────┐
        │  sandbox/            │                              │  data/                   │
        │  read-only SQL,      │                              │  pandas ground truth,    │
        │  3 defence layers    │                              │  task templates          │
        └──────────┬───────────┘                              └─────────────┬────────────┘
                   │                                                        │
                   │            ┌───────────────────────┐                   │
                   └───────────►│  env/  episode loop   │◄──────────────────┘
                                │  3 tools, call budget │
                                └───────────┬───────────┘
                                            │ Trajectory
                                ┌───────────▼───────────┐
                                │  rubric/  reward      │
                                │  named components     │
                                └───────────────────────┘
```

## Quick start

```bash
pip install -e '.[dev]'
python data/download_chinook.py     # one-time setup, ~1 MB
pytest                              # 166 tests
```

Run a real model through the environment:

```bash
export OPENAI_API_KEY=...
python eval/run_eval.py --model gpt-4o-mini --num-tasks 20

# or any OpenAI-compatible endpoint (vLLM, Together, OpenRouter, ...)
python eval/run_eval.py --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1 --num-tasks 50
```

## The data

[Chinook](https://github.com/lerocha/chinook-database) — a digital-media-store
sample database built from a real iTunes library. It is downloaded once as a
setup step, not regenerated per episode.

### Table names are not what you might expect

The official `Chinook_Sqlite.sqlite` build uses **singular PascalCase** names.
A widely-circulated community repackaging uses plural snake_case, and the two
are easy to confuse:

| Common assumption | Actual table in this build |
| --- | --- |
| `customers` | `Customer` |
| `invoices` | `Invoice` |
| `invoice_items` | **`InvoiceLine`** (PK `InvoiceLineId`) |
| `tracks`, `albums`, `artists`, `genres` | `Track`, `Album`, `Artist`, `Genre` |

Row counts (asserted in `tests/test_data_and_ground_truth.py`):
`Album` 347, `Artist` 275, `Customer` 59, `Employee` 8, `Genre` 25,
`Invoice` 412, `InvoiceLine` 2240, `MediaType` 5, `Playlist` 18,
`PlaylistTrack` 8715, `Track` 3503.

`Playlist`/`PlaylistTrack` carry no revenue signal and no template uses them.

### Schema (the parts that matter)

```
Customer(CustomerId PK, FirstName, LastName, Company, Address, City, State,
         Country, PostalCode, Phone, Fax, Email, SupportRepId → Employee)
Invoice(InvoiceId PK, CustomerId → Customer, InvoiceDate, BillingAddress,
        BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
InvoiceLine(InvoiceLineId PK, InvoiceId → Invoice, TrackId → Track,
            UnitPrice, Quantity)
Track(TrackId PK, Name, AlbumId → Album, MediaTypeId, GenreId → Genre,
      Composer, Milliseconds, Bytes, UnitPrice)
Album(AlbumId PK, Title, ArtistId → Artist)
Artist(ArtistId PK, Name)
Genre(GenreId PK, Name)
Employee(EmployeeId PK, LastName, FirstName, Title, ReportsTo → Employee, ...)
```

The revenue fact path is
`Customer → Invoice → InvoiceLine → Track → Album → Artist`.

### Facts discovered from the data, not assumed

Nothing about years, countries or genres is hardcoded — upstream Chinook shifts
its invoice dates forward periodically, so any literal year would rot. The task
generator reads them from the database at runtime.

- **Invoice date range: `2021-01-01` → `2025-12-22`** — five complete years,
  ~83 invoices each.
- **`Quantity` is always 1** and `UnitPrice ∈ {0.99, 1.99}`. Revenue is
  therefore coarse-grained, which makes **ties pervasive**: in 2021, *eight*
  customers tie for 5th place. See [Ties](#ties-are-real-and-graded-as-such).
- **50 of 59 customers have no invoices in at least one year.** This is what
  makes the year-over-year template a real test: a naive inner join silently
  drops exactly the customers whose revenue went to zero.
- **110 of 275 artists never sold anything**, which is what the anti-join
  template is built on.
- 24 countries, 9 of them with ≥2 customers.

### Canonical revenue definition

Revenue is **`SUM(InvoiceLine.UnitPrice * Quantity)`**.

Chinook offers a second route — `SUM(Invoice.Total)` — and this repo
cross-checks them. **They agree exactly**: all 412 invoices match to the cent,
grand total 2328.60 both ways. No data-quality finding.
(`test_two_revenue_definitions_agree`)

The line-item source is canonical because it is more granular and because it
forces the agent through a real join instead of reading a pre-aggregated
column. Currency is rounded **half-up** to cents at the end of an aggregation —
Python's built-in `round` is banker's rounding, which would disagree with what
a SQL engine reports.

`Invoice.BillingCountry == Customer.Country` on every row, so country questions
are unambiguous; `Customer.Country` is the documented choice.

## Task templates

Diversity has to come from the task layer, because the data is fixed. Each
template samples its parameters from values found in the database, so the agent
cannot memorise a single question→answer pair.

| Template | Tier | Answer type |
| --- | --- | --- |
| `customer_count_in_country` | single_table | numeric |
| `total_revenue_in_year` | single_table | numeric |
| `tracks_sold_in_genre_year` | single_table | numeric |
| `top_customers_by_revenue` | multi_join | ranked_list |
| `top_artists_by_revenue` | multi_join | ranked_list |
| `top_genre_in_country` | multi_join | categorical |
| `top_countries_by_revenue` | multi_join | ranked_list |
| `top_customer_revenue_increase` | time_comparison | ranked_list |
| `artists_with_no_sales_in_year` | time_comparison | entity_set |

Tiers are defined by how much joining and reshaping the question needs:
`single_table` (one aggregate), `multi_join` (3–5 table join through the fact
path), `time_comparison` (a join *plus* a two-window comparison or an
anti-join).

A parameterisation that would produce a degenerate task — an empty answer,
fewer entities than the requested K, an answer set so large it becomes a
transcription exercise — is rejected and resampled.

### Ground truth is computed in pandas, never from SQL

Every `compute_ground_truth` is a pandas aggregation over the raw tables.
**The reward function never treats "matches some reference SQL's output" as
ground truth**, because that only tests SQL-equivalence to one particular
solution, not correctness.

SQL does appear in `tests/test_data_and_ground_truth.py` — but only to *audit
the grader in a test*, never to define correctness at runtime.

### Edge cases, and how they are graded

**Zero-revenue periods.** A customer with no invoices in a period earns 0.00
for that period, not NULL. The year-over-year template reindexes over all
customers and fills, so customers absent from one year stay rankable.

**Rounding.** Half-up to cents, applied at the end of an aggregation. Numeric
answers are compared with `abs_tol=0.01` / `rel_tol=1e-3`; integer counts use
exact comparison.

#### Ties are real, and graded as such

The original plan was to define a tie-breaking rule. That turned out to be the
wrong answer for this dataset.

With only two possible unit prices, ties are not an edge case — they are the
common case. A "top 5 customers in 2021" question has eight customers tied at
5th place. Two obvious fixes both have real costs:

- *Reject tied parameterisations* — shrinks the task space substantially.
- *Append "break ties by CustomerId ascending"* — unnatural phrasing that tests
  instruction-following rather than analysis.

Instead, ranked ground truth is stored as an **ordered sequence of tie groups**,
and any linearisation consistent with the group order scores 1.0. If positions
5–12 all tie at $15.84, a top-5 answer may name *any one* of those eight
customers in position 5 and be fully correct. This is the semantics an analyst
would actually accept.

Ground truth renders with explicit rank bands:

```
1-3. Leonie Köhler | Tim Goyer | Dominique Lefebvre (tied)
4.   John Gordon
5-12. Alexandre Rocha | Jennifer Peterson | Julia Barnett | Aaron Mitchell
    | Hannah Schneider | Stanisław Wójcik | Phil Hughes | Luis Rojas (tied)
```

Grading by answer type:

| Type | Rule |
| --- | --- |
| `ranked_list` | Fraction of the K positions filled by an entity from the correct tie group. Answers truncate to K; a repeated name counts once. |
| `entity_set` | **Jaccard** similarity, not recall — so dumping the whole catalogue scores near zero rather than 1.0. |
| `numeric` | Absolute/relative tolerance; ambiguous text (two different numbers) scores 0. |
| `categorical` | Exact match against the winner(s). Naming more candidates than there are genuine ties is hedging and scores 0. A prose fallback accepts "The top genre is Rock." only when exactly one candidate from the whole universe is mentioned. |

Entity matching folds case, accents and punctuation (`Helena Holý` ≡
`Helena Holy`) and accepts the numeric primary key as an alias.

## The agent loop

Three tools, a budget of **12 tool calls**:

| Tool | Behaviour |
| --- | --- |
| `inspect_schema()` | Tables, columns, keys, row counts. |
| `execute_sql(query)` | One read-only statement; rows or a structured error. |
| `submit_answer(answer)` | Ends the episode. |

Budget exhaustion terminates **gracefully** — the trajectory is complete and
gradeable, with no answer and therefore no correctness credit. Nothing raises.
Malformed arguments, unknown tool names and calls after termination all come
back as explanatory observations, because a crash mid-rollout would destroy the
episode.

### Mechanics and reward are kept separate

The environment never refuses a legal-but-unwise action. `submit_answer` is
accepted even with no query run. Policing that in the environment would *hide*
the behaviour from the reward function, and shaping it is the whole point. So
the environment records; the rubric decides what it was worth. This is also
what makes adversarial trajectories constructible in tests.

### SQL sandbox: three independent defence layers

1. **Lexical screening** (`sandbox/safety.py`) — allow-list of leading keywords
   (`SELECT`/`WITH`/`EXPLAIN`/`VALUES`), forbidden-keyword scan with string
   literals blanked out, single-statement enforcement. This layer exists for
   *observation quality* — the agent should read "DROP is not allowed, this
   database is read-only", not "attempt to write a readonly database". It is
   explicitly **not** trusted for security.
2. **Read-only connection** — `file:...?mode=ro&immutable=1` plus
   `PRAGMA query_only=ON`.
3. **`sqlite3` authorizer** — denies every action except reads.

Layers 2 and 3 are what actually make an escape impossible;
`test_read_only_connection_rejects_writes_even_without_static_screen` proves
layer 1 is not load-bearing.

Plus: a **5s execution timeout** via progress handler, and a **50-row limit**
with honest truncation reporting.

`CASE ... END` and `replace()` are deliberately *not* in the forbidden-keyword
list — they are ordinary analytical SQL, and blocking them would break
legitimate queries while adding nothing (layers 2 and 3 already cover
transactions and mutation).

## Reward function

A rubric of named components summing to a scalar. **Every component is logged
separately**, so a training run can be debugged by looking at which term moved.

| Component | Weight | What it measures |
| --- | --- | --- |
| `correctness` | **+1.00** | Match to pandas ground truth over the full dataset |
| `grounding` | **gate** | Multiplies correctness by 0 or 1 |
| `schema_first_bonus` | +0.05 | One-time, for inspecting before the first query |
| `wasted_call_penalty` | −0.15 | −0.03 per information-free call, capped at 5 |
| `efficiency_penalty` | −0.10 | −0.02 per call past 8, capped at 5 |
| `no_query_attempt_penalty` | −0.20 | Charged when `execute_sql` was never called |

Total ∈ **[−0.45, +1.05]**. Positive shaping is at most **0.05 — 5% of the
correctness weight**. These are nudges, not the objective, and
`test_process_shaping_cannot_outweigh_correctness` asserts it rather than
merely claiming it.

Each component is a **pure function of `(Trajectory, GroundTruth)`** — no
database, no model, no framework — so all of them are unit-testable in
isolation.

### Why grounding is a gate, not an addend

As an additive penalty, grounding would be *tradeable*: an agent could eat the
loss and still collect correctness reward for an answer it never verified. As a
multiplier, an ungrounded answer is worth exactly zero no matter how right it
is. That is the property the design actually needs.

Grounding requires both:

**(a) A real query ran.** At least one successful `execute_sql` whose execution
actually read a data table — decided by **SQLite's own authorizer callback**,
which reports each table it resolves while compiling the statement. This is not
regex table-name matching, so it cannot be fooled by a table name inside a
string literal. `SELECT 'Acme Corp'` and `SELECT * FROM sqlite_master` both
fail it: they execute fine but read no data.

**(b) The answer appears in what came back.** At least one submitted value
occurs in a result row the agent actually saw, matched against the row's
concatenated text so a name split across `FirstName`/`LastName` still counts.

Condition (b) is deliberately lenient — *at least one* value, not all. Results
truncate at 50 rows, so requiring every submitted value would fail honest
agents whose answer sits below the cut. Correctness is the primary defence
against guessing (landing a ranked top-5 by chance out of 59 customers is a
1-in-5-million event); grounding is the backstop that stops an ungrounded
answer from ever collecting that reward.

### Anti-reward-hacking

Each mode below has a dedicated test in `tests/test_rubric.py`.

| Hack | Prevention | Test |
| --- | --- | --- |
| **Literal laundering** — `SELECT 'Acme Corp'` to satisfy "a query ran" | Grounding uses the SQLite authorizer's record of tables actually resolved, not the SQL text | `test_literal_laundering_is_gated_to_zero` |
| **Catalogue browsing** — `SELECT * FROM sqlite_master` | `sqlite_%` reads are tracked separately and never count as data reads | `test_catalogue_browsing_does_not_satisfy_grounding` |
| **Schema farming** — repeat `inspect_schema()` for a repeated bonus | Bonus fires once, and only before the first query; repeat inspections are charged as wasted calls | `test_schema_farming_earns_the_bonus_only_once`, `test_farming_is_worse_than_a_single_inspection` |
| **Cherry-picked / truncated results** | Correctness grades against ground truth recomputed over the full data; truncation is reported to the agent but never grades | `test_answer_matching_a_truncated_result_is_still_graded_in_full` |
| **Answer dumping** — list all 59 customers | Ranked answers truncate to K; sets use Jaccard, not recall | `test_dumping_every_customer_does_not_score`, `test_set_uses_jaccard_so_dumping_everything_scores_near_zero` |
| **Error-spam gaming** | Wasted-call penalty capped at −0.15; never attempting costs −0.20 | `test_never_querying_scores_worse_than_querying_and_failing` |
| **`SELECT 1` call-burning** | Successful-but-degenerate queries are charged like errors | `test_degenerate_queries_are_charged_like_errors` |
| **Categorical hedging** — "Rock or Latin or Jazz" | Naming more candidates than there are ties scores 0 | `test_categorical_rejects_hedging_across_several_candidates` |

Two of these came out of building it, not from the original checklist:

**The penalty-stacking bug.** `no_query_attempt_penalty` is keyed on
*attempts*, not successes. Keyed on successes, it would stack with the error
penalty and make an agent that tried five times and failed (−0.35) score
*worse* than one that never tried (−0.20) — inverting the exact incentive the
checklist asks for. Keyed on attempts: five failures cost −0.15, doing nothing
costs −0.20, so attempting always wins.

**Free schema farming.** Capping the bonus made farming unprofitable but not
*loss-making*: six inspections plus a query plus a submit is exactly the 8 free
calls, scoring identically to inspecting once. Redundant inspections are now
charged as wasted calls — the schema is immutable within an episode, so a
repeat returns byte-identical text, exactly like a successful `SELECT 1`.

### Worked examples

```
ideal        inspect → query → correct answer          reward +1.050
self-correct inspect → 2 errors → query → correct      reward +0.990
laundered    SELECT 'Hugh O''Reilly' → correct answer  reward -0.030  [GATED]
guessed      correct answer, no query at all           reward -0.200  [GATED]
farmed       9 × inspect_schema, no query              reward -0.320  [GATED]
```

The "guessed" row is the one that matters: the answer is **exactly right** and
still scores negative, because it was never grounded in a query.

## Framework integration

`verifiers` 0.3.1 ships both a `v1` stack and the classic v0 API. The adapter in
`sql_agent_rl/env/verifiers_env.py` targets the classic API
(`vf.StatefulToolEnv`, `vf.Rubric`), which is what the Environments Hub
`load_environment()` convention and the TRL/prime-rl integrations expect. **It
fits this design well** — no alternative framework was needed.

Two API details worth noting:

- `StatefulToolEnv.__init__(tools=...)` does *not* apply `args_to_skip`, so the
  tools are registered with explicit `add_tool(fn, args_to_skip=["episode"])`
  calls. This keeps the live episode handle out of the schema the model sees.
- Termination is a `@vf.stop` predicate rather than an `is_completed` override,
  so it composes with the base class's own stop conditions instead of
  pre-empting them.

**The core is framework-independent by design.** The sandbox, episode loop and
rubric are plain Python with no `verifiers` import; `verifiers` is an optional
extra. `eval/run_eval.py` drives the environment directly over any
OpenAI-compatible API, which both keeps the eval dependency-light and
demonstrates the portability. Moving to Gymnasium or OpenEnv would mean writing
a second adapter, not rewriting the environment.

```python
from sql_agent_rl.env.verifiers_env import load_environment

env = load_environment(num_tasks=500, seed=0, difficulties=["multi_join"])
```

## Known limitations

**Chinook is a very common SQL tutorial dataset.** A model may well have seen
its schema and typical query patterns during pretraining. That is acceptable
for a portfolio project, but it means **strong performance here is a weaker
generalisation signal** than performance on a schema the model has never seen.
The grounding gate mitigates the most direct form of this — an answer recalled
from pretraining rather than queried scores zero — but it cannot correct for a
model that simply finds this schema easier than an unfamiliar one. Treat
absolute scores as a sanity check on the environment, not as evidence of
general text-to-SQL ability.

**The dataset is small.** 59 customers and 412 invoices means aggregates are
coarse and ties are frequent. Handled explicitly in grading, but it limits how
finely tasks can discriminate.

**Grounding condition (b) is lenient by construction.** Requiring only one
traceable value avoids false negatives from the 50-row truncation, at the cost
of some strictness. An agent that ran a broad query and then guessed from it
could pass (b) — correctness is what stops that paying off.

**No live-model numbers are published here.** The eval harness is tested
end-to-end against a stubbed client (`tests/test_eval_harness.py`), but this
repo ships no measured baseline; run `eval/run_eval.py` to produce one.

## Repo layout

```
data/download_chinook.py        one-time dataset setup (runnable shim)
sql_agent_rl/
  data/    download_chinook.py  canonical downloader + fetch() for fixtures
           chinook.py           pandas frames, canonical revenue, rounding
           tasks.py             9 parameterised templates + ground truth
           answers.py           answer types, parsing, tie-group grading
  sandbox/ safety.py            lexical screening (layer 1)
           executor.py          read-only execution, authorizer, limits
  env/     trajectory.py        the record the rubric grades
           episode.py           tool loop, budget (framework-independent)
           verifiers_env.py     verifiers adapter (optional import)
  rubric/  components.py        pure reward functions
           reward.py            weights, gate, breakdown
eval/run_eval.py                run a real model, report per-component stats
tests/                          166 tests
```

## Test suite

```
tests/test_sandbox_safety.py         60  destructive statements, limits, timeouts
tests/test_answer_grading.py         33  parsing, tie groups, Jaccard, tolerances
tests/test_rubric.py                 21  adversarial trajectories, reward bounds
tests/test_data_and_ground_truth.py  15  row counts, revenue cross-check, determinism
tests/test_episode.py                14  tool loop, budget, graceful termination
tests/test_verifiers_env.py          14  adapter: schemas, state, rubric parity
tests/test_eval_harness.py            9  eval loop against a stubbed model
                                    ---
                                    166
```

## Status

Steps 1–6 of the build are complete: data layer, sandbox, task templates and
ground truth, environment loop, reward rubric, and the `verifiers` wrapper plus
eval harness. Wiring into `prime-rl` or TRL for an actual GRPO run is the
remaining stretch goal.
