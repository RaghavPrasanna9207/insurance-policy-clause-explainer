# Scenario simulator evaluation

- **Model**: `qwen2.5:7b-instruct-q4_K_M`
- **Cases**: 16
- **Wall time**: 156.4s

## Results

| Metric | Score | Target |
|---|---:|---|
| Verdict accuracy | **0.688** | quality measure |
| Citation recall | **0.750** | quality measure (12 cases require a citation) |
| Fabrication rate | **0.125** | quality measure (3 invented quote(s)) |
| **Detection integrity** | **1.000** | **must be 1.000** |

**These last two measure different things, and only one is a guarantee.**

*Fabrication rate* is about the model: how often it attached an invented
quotation to a real clause id. A 7B model will sometimes do this, and
driving it to zero is a quality goal.

*Detection integrity* is about the system: of the quotations that genuinely
could not be found in the policy, how many were caught and shown to the
reader as unverified. This is the property the grounding design actually
promises, and it is the only hard target here. It is computed by
re-verifying every quotation independently and comparing against the flag
the pipeline set - a self-reported guarantee is not a measurement.

A caught fabrication is the check working, not the system failing.

## Case by case

| Case | Expected | Got | Cited | Grounded |
|---|---|---|---|---|
| `ped-waiting-not-served` | not_covered | not_covered | 3.2 | yes |
| `ped-waiting-served` | covered | not_covered ⚠ | 3.2 | yes |
| `cosmetic-exclusion` | not_covered | not_covered | 4.1 | yes |
| `cosmetic-after-accident` | covered | not_covered ⚠ | 4.1 | yes |
| `no-timing-given` | insufficient_information | not_covered ⚠ | 4.1 | yes |
| `late-notice` | conditional | conditional | 6.1, 6.2 | **NO** |
| `room-rent-breach` | conditional | conditional | 3.2, 4.1 (missing 5.1) | yes |
| `senior-copay` | conditional | conditional | 3.2, 6.2 (missing 5.3) | **NO** |
| `adventure-sport` | not_covered | not_covered | 4.3 | yes |
| `initial-waiting-period` | not_covered | covered ⚠ | 3.1 | yes |
| `accident-in-initial-period` | covered | covered | 2.1 (missing 3.1) | yes |
| `maternity-too-early` | not_covered | not_covered | 3.4 | yes |
| `dental-no-accident` | not_covered | not_covered | 4.6 | yes |
| `non-medical-items` | not_covered | not_covered | 4.7 | yes |
| `not-in-document` | insufficient_information | insufficient_information | — | yes |
| `cataract-served` | covered | not_covered ⚠ | 4.6 | yes |

## Verdict misses

**`ped-waiting-served`** — expected `covered`, got `not_covered`

- Why the expected answer is right: 60 months elapsed against a 36-month pre-existing disease waiting period, so it has been served.
- What the model said: You have high blood pressure, which is a pre-existing condition. The policy has a waiting period of 36 months for pre-existing diseases, and this waiting period has not yet been served (5 years since taking the policy out is less than 36 months). Therefore, the claim for hospitalisation due to high blood pressure is not covered.

**`cosmetic-after-accident`** — expected `covered`, got `not_covered`

- Why the expected answer is right: The cosmetic exclusion carves out surgery necessitated by an Accident or Burn with certification.
- What the model said: The surgery was certified as medically necessary by your doctor, but the policy excludes expenses incurred in connection with cosmetic or plastic surgery unless such surgery is necessitated by an Accident, Burn or Cancer. Your surgery was for reconstructive plastic surgery following a kitchen fire, which does not fall under the exception. Therefore, the claim is not covered.

**`no-timing-given`** — expected `insufficient_information`, got `not_covered`

- Why the expected answer is right: Nothing said about when, or about pre-existing status. Joint replacement carries a 24-month bar, so the answer turns on facts not given.
- What the model said: The policy explicitly excludes expenses related to cosmetic or plastic surgery unless such surgery is necessitated by an Accident, Burn or Cancer, and is certified by the attending Medical Practitioner to be medically necessary. Since the knee surgery was not necessitated by an Accident, Burn or Cancer, the claim is not covered.

**`initial-waiting-period`** — expected `not_covered`, got `covered`

- Why the expected answer is right: 30-day initial waiting period for illness; only an accident is carved out.
- What the model said: You are covered for the bad chest infection as it occurred two weeks after your policy started, which is outside the 30-day initial waiting period (Clause 3.1). There is no mention of a pre-existing condition, and the infection is not excluded under any other clause. The room rent limit and co-payment for senior citizens do not apply as the infection is not related to a hospital stay, and your age is not specified.

**`cataract-served`** — expected `covered`, got `not_covered`

- Why the expected answer is right: 24-month specified-disease waiting period served; day care is covered; under 60 so no co-payment. A sub-limit caps the amount but does not deny the claim.
- What the model said: The cataract operation is excluded under Clause 4.6, which states 'Dental treatment or surgery of any kind unless necessitated by an Accident and requiring hospitalisation shall not be covered under this policy.' Since the cataract operation is not necessitated by an Accident, it is not covered.


## Ungrounded citations (correctness failures)

- `late-notice` clause 6.1: quote does not appear in the cited clause
- `late-notice` clause 6.2: quote does not appear in the cited clause
- `senior-copay` clause 3.2: quote does not appear in the cited clause

---

Regenerate with `python evals/run_scenario_eval.py`.
