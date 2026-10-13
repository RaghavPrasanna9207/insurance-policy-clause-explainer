"""Does any text rule give a different answer when only the FORMAT of its input changes?

Every compiled pattern in the pipeline is run over every clause of the three
policies and every eval question, once on the text as stored and once on the
same text with only its typesetting changed (line breaks to spaces, a hyphen
at a line end joined, typographic characters to ASCII; for questions, curly
apostrophes, capitals, doubled spaces). A different number of matches means
the rule is reading the layout and not the words.

A line of output is a lead, not a verdict: some patterns are applied to text
that is already normalised, and a few use layout on purpose (the reading
grade treats a line as a unit when a text has no full stops). Each has to be
checked where it is used. M45 in the learning log walks through the first run.

No model calls: the clause analysis replays from the cache. Needs the real
policy PDFs (python evals/real/fetch_policies.py).

Usage:
    python evals/check_format.py
"""
import asyncio, glob, importlib, json, re, sys, unicodedata
from pathlib import Path
sys.path.insert(0, "evals"); sys.path.insert(0, "api")
from run_scenario_eval import build_clauses

MODS = ["app.grounding", "app.pipeline.analyze", "app.pipeline.reduction", "app.pipeline.scenario",
        "app.pipeline.score", "app.pipeline.waiting", "app.pipeline.window", "app.pipeline.segment",
        "app.pipeline.ingest", "app.llm.prompts"]
PATS = {}
for name in MODS:
    mod = importlib.import_module(name)
    for attr, val in vars(mod).items():
        if isinstance(val, re.Pattern):
            PATS[f"{name.split('.')[-1]}.{attr}"] = val

def collapse(t): return " ".join(t.split())
def dehyphen(t): return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
ASCII = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2010": "-", "\u2011": "-",
                       "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u00ad": ""})
def ascii_punct(t): return unicodedata.normalize("NFKC", t).translate(ASCII)
CLAUSE_T = {"line breaks -> spaces": collapse, "line-end hyphen joined": dehyphen, "typographic -> ASCII": ascii_punct}

def curly(t): return t.replace("'", "\u2019")
def upper(t): return t.upper()
def wide(t): return t.replace(" ", "  ")
QUESTION_T = {"curly apostrophes": curly, "ALL CAPS": upper, "double spaces": wide}

def diff_table(texts, transforms, label):
    print(f"\n######## {label}: {len(texts)} texts")
    for tname, f in transforms.items():
        changed_inputs = sum(1 for _, t in texts if f(t) != t)
        print(f"\n--- {tname}  (changes {changed_inputs} of the texts)")
        for pname, pat in sorted(PATS.items()):
            rows = [(cid, len(pat.findall(t)), len(pat.findall(f(t)))) for cid, t in texts]
            bad = [(c, a, b) for c, a, b in rows if a != b]
            if bad:
                print(f"   {pname:<42} differs on {len(bad):>3} texts   e.g. {bad[0][0]}: {bad[0][1]} -> {bad[0][2]}")

async def main():
    clause_texts = []
    feats = {"line-end hyphenation": r"[a-z]-\n\s*[a-z]", "U+2010/2011 hyphen": "[\u2010\u2011]", "en/em dash": "[\u2013\u2014]",
             "soft hyphen": "\u00ad", "curly quote": "[\u2018\u2019\u201c\u201d]", "nbsp": "\u00a0",
             "(30) days style": r"\(\d+\)\s*(?:days|months|years)", "30-day style": r"\d+-(?:day|month|year)",
             "ligature": "[\ufb00-\ufb06]", "tab/control": r"[\x00-\x08\x0b-\x1f]"}
    for pdf in ["evals/golden/synthetic-health-policy.pdf", "samples/real/star-arogya-sanjeevani.pdf", "samples/real/hdfc-optima-secure.pdf"]:
        cl = await build_clauses(Path(pdf))
        short = Path(pdf).stem[:5]
        clause_texts += [(f"{short}:{c.clause_id}", c.text) for c in cl]
        joined = "\n\n".join(c.text for c in cl)
        print(f"{short}: " + ", ".join(f"{k}={len(re.findall(v, joined))}" for k, v in feats.items()))
    diff_table(clause_texts, CLAUSE_T, "CLAUSE TEXT")
    qs = []
    for f in glob.glob("evals/golden/scenarios*.json") + glob.glob("evals/real/scenarios*.json"):
        qs += [(c["id"], c["scenario"]) for c in json.load(open(f, encoding="utf-8"))["cases"]]
    diff_table(qs, QUESTION_T, "QUESTIONS")

asyncio.run(main())
