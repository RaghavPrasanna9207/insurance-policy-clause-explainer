# Evaluation report

- **Model**: `qwen2.5:7b-instruct-q4_K_M`
- **Prompt version**: see `PROMPT_VERSION` in `api/app/llm/prompts.py`
- **Clauses**: 39/39 analysed
- **Wall time**: 0.0s

## Classification

**Macro-F1: 1.000** &nbsp;&nbsp; Accuracy: 1.000

Macro-F1 is the headline: it weights every clause type equally, so the
4 waiting periods count as much as the 8 exclusions. Accuracy would let
a model coast on the common types.

| Clause type | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `coverage` | 6 | 1.00 | 1.00 | 1.00 |
| `exclusion` | 8 | 1.00 | 1.00 | 1.00 |
| `condition` | 6 | 1.00 | 1.00 | 1.00 |
| `sub_limit` | 5 | 1.00 | 1.00 | 1.00 |
| `waiting_period` | 4 | 1.00 | 1.00 | 1.00 |
| `definition` | 5 | 1.00 | 1.00 | 1.00 |
| `procedural` | 5 | 1.00 | 1.00 | 1.00 |

## Ranking quality

**6/9 expectations met** (top-10)

Classification F1 measures labelling. This measures the ranking, which
is the actual product: a system can label every clause perfectly and
still bury the room-rent cap beneath a harmless administrative power.
Expectations are hand-authored in `golden/ranking-expectations.json`
from how claims actually go wrong, written before any ranking was seen.

### Should be in the top 10, but is not

| Clause | Actual rank | Why it matters |
|---|---:|---|
| `3.2` | 11 | 36-month pre-existing disease waiting period. Affects most buyers over 40, and is what people most often assume they are covered for. |
| `5.3` | 21 | 20% senior-citizen co-payment. Applies to every claim for a large class of policyholder, permanently. |

### In the top 10, but should not be

| Clause | Actual rank | Why it does not belong |
|---|---:|---|
| `6.5` | 9 | The Company may require a medical examination AT ITS OWN EXPENSE. A power the insurer must pay to exercise is not a financial risk to the policyholder. |


## Top 10 by impact score

This is the app's actual output: the clauses most likely to cost a
policyholder money, ranked. `buriedness` is computed, not model-judged.

| # | Clause | Type | Impact | Buried | Grade |
|---|---|---|---:|---:|---:|
| 1 | 4.7 Non-Medical Expenses | `exclusion` | 85.0 | 0.47 | 16 |
| 2 | 6.1 Notice of Claim | `condition` | 84.7 | 0.46 | 16 |
| 3 | 6.2 Submission of Documents | `condition` | 83.2 | 0.38 | 13 |
| 4 | 6.3 Disclosure of Material Facts | `condition` | 82.3 | 0.33 | 14 |
| 5 | 5.2 Proportionate Deduction | `sub_limit` | 65.3 | 0.39 | 13 |
| 6 | 6.6 Contribution | `condition` | 62.0 | 0.35 | 14 |
| 7 | 5.1 Room Rent Limit | `sub_limit` | 57.1 | 0.48 | 14 |
| 8 | 4.5 Infertility and Sterility | `exclusion` | 52.9 | 0.45 | 22 |
| 9 | 6.5 Medical Examination | `condition` | 52.8 | 0.44 | 17 |
| 10 | 4.1 Cosmetic and Plastic Surgery | `exclusion` | 51.9 | 0.37 | 18 |

---

Regenerate with `python evals/run_eval.py`.
