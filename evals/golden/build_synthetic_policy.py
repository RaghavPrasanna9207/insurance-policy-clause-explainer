"""Generate a synthetic IRDAI-style health policy PDF plus its ground-truth labels.

WHY THIS EXISTS
---------------
Every stage of this pipeline needs something to be measured against. Real
insurer policy wordings are unsuitable for that job for two reasons:

1. Licensing. Redistributing an insurer's copyrighted wording in a public repo
   is not something we can do, so the repo could not be self-contained.
2. Ground truth. Nobody has labelled them. Hand-labelling 40 clauses of real
   legal text is slow and produces labels of uncertain quality.

So we author the policy ourselves. Because the PDF and the label file are
generated from the SAME source of truth (the CLAUSES list below), they can never
disagree - there is no separate labelling step to drift out of sync.

The clauses are written to imitate real IRDAI health policy wordings closely,
including the deliberately awkward cases that break naive classifiers:
  - a waiting period phrased to sound exactly like an exclusion
  - a sub-limit phrased to sound exactly like a benefit
  - a definition ("Hospital") whose narrowness quietly guts coverage elsewhere

Run:  python evals/golden/build_synthetic_policy.py
Emits: synthetic-health-policy.pdf  and  synthetic-health-policy.labels.json
"""

import json
from pathlib import Path

import fitz  # PyMuPDF

OUT_DIR = Path(__file__).parent
PDF_PATH = OUT_DIR / "synthetic-health-policy.pdf"
LABELS_PATH = OUT_DIR / "synthetic-health-policy.labels.json"
HOSTILE_PATH = OUT_DIR / "synthetic-hostile-policy.pdf"
MINI_PATH = OUT_DIR / "synthetic-mini-policy.pdf"

# --- Page layout constants -------------------------------------------------
PAGE_W, PAGE_H = fitz.paper_size("a4")
MARGIN = 56
BODY_W = PAGE_W - 2 * MARGIN

FONT_BODY, SIZE_BODY = "helv", 9.5
FONT_BOLD, SIZE_HEAD = "hebo", 13
SIZE_SUBHEAD = 10.5

# --- The policy itself -----------------------------------------------------
# ("SECTION", title) starts a new section.
# (number, heading, text, expected_clause_type) is a clause.
#
# expected_clause_type uses the taxonomy in api/app/taxonomy.py.

CLAUSES: list[tuple] = [
    ("SECTION", "SECTION 1 - DEFINITIONS"),
    (
        "1.1", "Hospital",
        "Hospital means any institution established for in-patient care and day care treatment of "
        "illness or injuries which has been registered as a hospital with the local authorities, and "
        "which has at least ten in-patient beds in towns having a population of less than ten lakhs "
        "and fifteen in-patient beds in all other places, qualified nursing staff under its employment "
        "round the clock, a fully equipped operation theatre of its own where surgical procedures are "
        "carried out, and which maintains daily records of patients accessible to the Company's "
        "authorised personnel.",
        "definition",
    ),
    (
        "1.2", "Pre-existing Disease",
        "Pre-existing Disease means any condition, ailment, injury or disease that is diagnosed by a "
        "physician within forty eight months prior to the effective date of the policy issued by the "
        "Insurer, or for which medical advice or treatment was recommended by, or received from, a "
        "physician within forty eight months prior to the effective date of the policy.",
        "definition",
    ),
    (
        "1.3", "Sum Insured",
        "Sum Insured means the pre-defined limit specified in the Policy Schedule which represents the "
        "maximum aggregate liability of the Company for any and all claims arising during the Policy "
        "Year in respect of each Insured Person.",
        "definition",
    ),
    (
        "1.4", "Day Care Treatment",
        "Day Care Treatment means medical treatment or surgical procedure which is undertaken under "
        "general or local anaesthesia in a Hospital or day care centre in less than twenty four hours "
        "because of technological advancement, and which would otherwise have required hospitalisation "
        "of more than twenty four hours.",
        "definition",
    ),
    (
        "1.5", "Reasonable and Customary Charges",
        "Reasonable and Customary Charges means the charges for services or supplies which are the "
        "standard charges for the specific provider and consistent with the prevailing charges in the "
        "geographical area for identical or similar services, taking into account the nature of the "
        "illness or injury involved.",
        "definition",
    ),

    ("SECTION", "SECTION 2 - SCOPE OF COVER"),
    (
        "2.1", "In-patient Hospitalisation",
        "The Company shall indemnify the Medical Expenses incurred by the Insured Person towards "
        "hospitalisation exceeding twenty four consecutive hours during the Policy Year for medically "
        "necessary treatment of an Illness or Injury contracted or sustained during the Policy Period, "
        "up to the Sum Insured specified in the Policy Schedule.",
        "coverage",
    ),
    (
        "2.2", "Pre-hospitalisation Medical Expenses",
        "The Company shall indemnify Medical Expenses incurred during the sixty days immediately "
        "preceding the date of admission, provided that such expenses relate to the same condition for "
        "which the Insured Person's hospitalisation was required and the in-patient claim has been "
        "accepted by the Company.",
        "coverage",
    ),
    (
        "2.3", "Post-hospitalisation Medical Expenses",
        "The Company shall indemnify Medical Expenses incurred during the ninety days immediately "
        "following the date of discharge, provided that such expenses relate to the same condition for "
        "which the Insured Person was hospitalised and the in-patient claim has been accepted.",
        "coverage",
    ),
    (
        "2.4", "Day Care Procedures",
        "The Company shall indemnify Medical Expenses incurred for Day Care Treatment listed in Annexure "
        "II which does not require twenty four hours of hospitalisation, provided such treatment is "
        "undertaken in a Hospital or a registered day care centre.",
        "coverage",
    ),
    (
        "2.5", "Road Ambulance",
        "The Company shall reimburse expenses incurred on transportation of the Insured Person by road "
        "ambulance to a Hospital for treatment following an Emergency, provided the in-patient claim is "
        "admissible under Clause 2.1.",
        "coverage",
    ),
    (
        "2.6", "AYUSH Treatment",
        "The Company shall indemnify Medical Expenses incurred for in-patient treatment taken under "
        "Ayurveda, Yoga, Naturopathy, Unani, Siddha and Homeopathy systems of medicine in a government "
        "hospital or in an institute accredited by the Quality Council of India.",
        "coverage",
    ),

    ("SECTION", "SECTION 3 - WAITING PERIODS"),
    (
        "3.1", "Initial Waiting Period",
        "No claim shall be payable in respect of any Illness contracted during the first thirty days "
        "from the commencement date of the first policy with the Company, except claims arising out of "
        "an Accident. This exclusion shall not apply on subsequent continuous renewal of the policy "
        "without a break.",
        "waiting_period",
    ),
    (
        "3.2", "Pre-existing Disease Waiting Period",
        "Expenses related to the treatment of a Pre-existing Disease and its direct complications shall "
        "be excluded until the expiry of thirty six months of continuous coverage after the date of "
        "inception of the first policy with the Company.",
        "waiting_period",
    ),
    (
        "3.3", "Specified Disease Waiting Period",
        "Expenses related to the treatment of cataract, hernia, hysterectomy, benign prostatic "
        "hypertrophy, joint replacement surgery, and diseases of the ear, nose and throat shall be "
        "excluded until the expiry of twenty four months of continuous coverage from the date of "
        "inception of the first policy.",
        "waiting_period",
    ),
    (
        "3.4", "Maternity Waiting Period",
        "Expenses related to maternity, childbirth and lawful medical termination of pregnancy shall be "
        "covered only after the Insured Person has been continuously covered for a period of thirty six "
        "months under this policy, and shall be limited to two deliveries during the lifetime of the "
        "Insured Person.",
        "waiting_period",
    ),

    ("SECTION", "SECTION 4 - PERMANENT EXCLUSIONS"),
    (
        "4.1", "Cosmetic and Plastic Surgery",
        "The Company shall not be liable to make any payment in respect of expenses incurred in "
        "connection with cosmetic or plastic surgery or any treatment to change appearance, unless such "
        "surgery is necessitated by an Accident, Burn or Cancer, and is certified by the attending "
        "Medical Practitioner to be medically necessary.",
        "exclusion",
    ),
    (
        "4.2", "Intentional Self-Injury",
        "The Company shall not be liable for expenses arising out of or attributable to any intentional "
        "self-injury, attempted suicide, or the use of intoxicating substances, alcohol or drugs not "
        "prescribed by a registered Medical Practitioner.",
        "exclusion",
    ),
    (
        "4.3", "Hazardous Activities",
        "Expenses related to any treatment necessitated due to participation in hazardous or adventure "
        "sports, including but not limited to para-jumping, rock climbing, mountaineering, scuba diving "
        "and bungee jumping, shall not be admissible under this policy.",
        "exclusion",
    ),
    (
        "4.4", "Breach of Law",
        "Expenses for treatment directly arising from or consequent upon any Insured Person committing "
        "or attempting to commit a breach of law with criminal intent shall be excluded.",
        "exclusion",
    ),
    (
        "4.5", "Infertility and Sterility",
        "Expenses related to sterility and infertility, including assisted reproduction services, "
        "gestational surrogacy, reversal of sterilisation and any form of contraception, are excluded "
        "under this policy.",
        "exclusion",
    ),
    (
        "4.6", "Dental Treatment",
        "Dental treatment or surgery of any kind unless necessitated by an Accident and requiring "
        "hospitalisation shall not be covered under this policy.",
        "exclusion",
    ),
    (
        "4.7", "Non-Medical Expenses",
        "The Company shall not pay for items of personal comfort and convenience including telephone, "
        "television, internet charges, attendant charges, admission or registration fees, and any other "
        "item listed in Annexure III to this policy.",
        "exclusion",
    ),
    (
        "4.8", "War and Nuclear Perils",
        "Any expense arising from war, invasion, act of foreign enemy, warlike operations, or from "
        "nuclear weapons material or ionising radiation, whether war be declared or not, is excluded.",
        "exclusion",
    ),

    ("SECTION", "SECTION 5 - LIMITS AND CO-PAYMENT"),
    (
        "5.1", "Room Rent Limit",
        "Expenses towards room rent, boarding and nursing charges shall be limited to one percent of the "
        "Sum Insured per day, and expenses towards Intensive Care Unit charges shall be limited to two "
        "percent of the Sum Insured per day, subject to the Reasonable and Customary Charges.",
        "sub_limit",
    ),
    (
        "5.2", "Proportionate Deduction",
        "Where the Insured Person is admitted to a room category whose rent exceeds the limit specified "
        "in Clause 5.1, all associated medical expenses including surgeon fees, anaesthetist fees and "
        "operation theatre charges shall be reduced in the same proportion as the eligible room rent "
        "bears to the actual room rent charged.",
        "sub_limit",
    ),
    (
        "5.3", "Co-payment for Senior Citizens",
        "All admissible claims in respect of an Insured Person who has completed sixty years of age at "
        "the time of first inception of the policy shall be subject to a co-payment of twenty percent of "
        "the admissible claim amount, to be borne by the Insured Person.",
        "sub_limit",
    ),
    (
        "5.4", "Cataract Sub-limit",
        "Expenses in respect of treatment of cataract shall be limited to twenty five thousand rupees "
        "per eye, or ten percent of the Sum Insured, whichever is lower, per Policy Year.",
        "sub_limit",
    ),
    (
        "5.5", "Modern Treatment Limit",
        "Expenses incurred on advanced treatment methods including robotic surgery, stem cell therapy, "
        "oral chemotherapy and deep brain stimulation shall be restricted to fifty percent of the Sum "
        "Insured per Policy Year.",
        "sub_limit",
    ),

    ("SECTION", "SECTION 6 - CONDITIONS PRECEDENT TO LIABILITY"),
    (
        "6.1", "Notice of Claim",
        "Written notice of any claim must be given to the Company or its authorised Third Party "
        "Administrator within twenty four hours of admission in the case of Emergency hospitalisation, "
        "and at least forty eight hours prior to admission in the case of Planned hospitalisation, "
        "failing which the Company may at its discretion repudiate the claim.",
        "condition",
    ),
    (
        "6.2", "Submission of Documents",
        "All supporting documents relating to a claim, including original bills, discharge summary, "
        "investigation reports and prescriptions, must be submitted to the Company within fifteen days "
        "of the date of discharge from the Hospital. Failure to furnish such documents within the "
        "stipulated period may result in rejection of the claim.",
        "condition",
    ),
    (
        "6.3", "Disclosure of Material Facts",
        "The policy shall be void and all premium paid shall be forfeited to the Company in the event of "
        "misrepresentation, mis-description or non-disclosure of any material fact by the Insured Person "
        "at the time of proposal or at any time thereafter.",
        "condition",
    ),
    (
        "6.4", "Pre-authorisation for Cashless",
        "Cashless facility shall be available only at Network Providers and only where the Company or "
        "its Third Party Administrator has issued a written pre-authorisation prior to the commencement "
        "of treatment. Denial of pre-authorisation does not by itself constitute denial of the claim.",
        "condition",
    ),
    (
        "6.5", "Medical Examination",
        "The Company shall be entitled to require the Insured Person to undergo a medical examination by "
        "a Medical Practitioner nominated by the Company, at the Company's expense, as often as may be "
        "reasonably required in connection with any claim under this policy.",
        "condition",
    ),
    (
        "6.6", "Contribution",
        "If at the time of a claim the Insured Person holds any other policy of indemnity covering the "
        "same risk, the Company shall not be liable to pay more than its rateable proportion of the "
        "claim, and the Insured Person shall disclose particulars of such other policy to the Company.",
        "condition",
    ),

    ("SECTION", "SECTION 7 - GENERAL PROVISIONS"),
    (
        "7.1", "Free Look Period",
        "The Insured Person may cancel this policy within fifteen days of receipt of the policy document "
        "by giving written notice to the Company, and shall be refunded the premium paid less any "
        "proportionate risk premium for the period of cover and the expenses incurred on medical "
        "examination and stamp duty.",
        "procedural",
    ),
    (
        "7.2", "Renewal",
        "This policy shall ordinarily be renewable for life, provided the policy is renewed within the "
        "Grace Period of thirty days from the date of expiry and the Company has not discontinued the "
        "product. Renewal shall not be denied on the ground of an adverse claims history alone.",
        "procedural",
    ),
    (
        "7.3", "Portability",
        "The Insured Person may port this policy to another insurer at the time of renewal by making an "
        "application to the new insurer at least forty five days before the renewal date, and shall be "
        "given credit for waiting periods already served under this policy.",
        "procedural",
    ),
    (
        "7.4", "Grievance Redressal",
        "Any grievance in connection with this policy may be addressed to the Grievance Redressal "
        "Officer of the Company. If the grievance remains unresolved for a period exceeding thirty days, "
        "the Insured Person may approach the Insurance Ombudsman appointed under the Insurance "
        "Ombudsman Rules.",
        "procedural",
    ),
    (
        "7.5", "Cancellation by the Company",
        "The Company may cancel this policy on grounds of established fraud by giving fifteen days "
        "written notice to the Insured Person at the address last recorded, and no premium shall be "
        "refunded in such an event.",
        "procedural",
    ),
]


class _Writer:
    """Flows blocks down the page, starting a new page when one will not fit.

    PyMuPDF has no automatic pagination, so we do it by hand: try to place the
    block in the remaining space, and if `insert_textbox` reports it did not
    fit (a negative return), start a fresh page and place it there.
    """

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.page = None
        self.y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN

    def write(self, text: str, font: str, size: int | float, gap_after: float) -> int:
        """Place `text`, paginating if needed. Returns the page number used."""
        for _ in range(2):
            # Guard against an inverted rectangle. After a block that nearly
            # fills a page, `self.y` plus the trailing gap can land below the
            # bottom margin, which makes the next Rect empty and PyMuPDF raise
            # "text box must be finite and not empty". Start a fresh page if
            # there is not even room for one line.
            if PAGE_H - MARGIN - self.y < size * 2:
                self._new_page()
            rect = fitz.Rect(MARGIN, self.y, PAGE_W - MARGIN, PAGE_H - MARGIN)
            # insert_textbox returns leftover vertical space, or a negative
            # number meaning "this did not fit". We first place it into a
            # throwaway measurement to learn the height, then place for real.
            leftover = self.page.insert_textbox(
                rect, text, fontname=font, fontsize=size, align=fitz.TEXT_ALIGN_LEFT
            )
            if leftover >= 0:
                self.y = PAGE_H - MARGIN - leftover + gap_after
                return self.page.number
            self._new_page()
        raise RuntimeError("block too large for a single page: " + text[:60])


def build_mini() -> None:
    """A 4-clause policy, for tests that must run the whole pipeline.

    The API tests need a real upload-to-results round trip, but analysis costs
    roughly three seconds per clause on a local model. Using the 39-clause
    golden policy would make a single test take two minutes, which is long
    enough that people stop running the suite.

    Four clauses covering four different types is enough to prove the pipeline
    wires together end to end. Measuring how WELL it classifies is the eval
    harness's job, on the full document - a distinction worth keeping: tests
    check that it works, evals check how well.
    """
    doc = fitz.open()
    w = _Writer(doc)

    w.write("MINI HEALTH POLICY", FONT_BOLD, 14, 8)
    w.write("SECTION 1 - COVER AND EXCLUSIONS", FONT_BOLD, SIZE_HEAD, 6)

    picks = [c for c in CLAUSES if c[0] in ("2.1", "3.2", "4.1", "5.1")]
    for number, heading, text, _ in picks:
        w.write(f"{number} {heading}", FONT_BOLD, SIZE_SUBHEAD, 2)
        w.write(text, FONT_BODY, SIZE_BODY, 10)

    doc.save(MINI_PATH)
    doc.close()
    print(f"wrote {MINI_PATH.name} ({len(picks)} clauses, for fast API tests)")


def build_hostile() -> None:
    """The same policy, typeset with NO structural signal whatsoever.

    WHY THIS EXISTS
    ---------------
    The main golden PDF is generated by this repo, and the segmenter's font-size
    thresholds were tuned against it. Testing only on that document would be
    circular: it would prove the segmenter handles documents we designed to be
    easy, and say nothing about real ones.

    So this variant strips every advantage away - one font, one size, no bold,
    no styling at all. The only remaining structural signals are the ones a real
    machine-generated policy PDF often leaves you with:

      - clause numbering at the start of a line ("4.2")
      - ALL-CAPS section names matching known policy vocabulary

    Those are exactly rules 3 and 4 in `segment.py`. This document is what
    proves they work, rather than merely existing as untested fallback code.
    """
    doc = fitz.open()
    w = _Writer(doc)

    flat_font, flat_size = "helv", 10.0
    w.write("MEDIGUARD COMPREHENSIVE HEALTH INSURANCE POLICY", flat_font, flat_size, 4)
    w.write("Synthetic test document. Not a real insurance policy.", flat_font, flat_size, 8)

    for entry in CLAUSES:
        if entry[0] == "SECTION":
            w.write(entry[1], flat_font, flat_size, 4)
            continue
        number, heading, text, _ = entry
        # Heading and body run together as one paragraph, the way a poorly
        # structured PDF presents them.
        w.write(f"{number} {heading}. {text}", flat_font, flat_size, 6)

    doc.save(HOSTILE_PATH)
    doc.close()
    print(f"wrote {HOSTILE_PATH.name} (uniform font, no styling)")


def build() -> None:
    doc = fitz.open()
    w = _Writer(doc)

    w.write("MEDIGUARD COMPREHENSIVE HEALTH INSURANCE POLICY", FONT_BOLD, 16, 6)
    w.write("Policy Wording - UIN: SYNTH0001V012026", FONT_BODY, 9, 4)
    w.write(
        "This is a synthetic document created for testing an automated policy analysis system. "
        "It is not a real insurance policy and confers no cover of any kind.",
        FONT_BODY, 8, 18,
    )

    labels = []
    section_title = ""
    for entry in CLAUSES:
        if entry[0] == "SECTION":
            section_title = entry[1]
            w.write(section_title, FONT_BOLD, SIZE_HEAD, 8)
            continue

        number, heading, text, expected = entry
        # Heading and body are written as separate blocks so that the heading
        # keeps its own font size and weight. That difference is exactly the
        # signal the segmenter uses to find clause boundaries.
        w.write(f"{number} {heading}", FONT_BOLD, SIZE_SUBHEAD, 2)
        page_no = w.write(text, FONT_BODY, SIZE_BODY, 10)

        labels.append({
            "number": number,
            "heading": heading,
            "expected_type": expected,
            "section": section_title,
            "page": page_no + 1,  # 1-indexed for humans
            "text": text,
        })

    doc.save(PDF_PATH)
    doc.close()

    LABELS_PATH.write_text(
        json.dumps(
            {
                "source": "synthetic, authored for this repository",
                "clause_count": len(labels),
                "clauses": labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_type: dict[str, int] = {}
    for label in labels:
        by_type[label["expected_type"]] = by_type.get(label["expected_type"], 0) + 1

    print(f"wrote {PDF_PATH.name}: {doc.page_count if not doc.is_closed else ''}")
    print(f"wrote {LABELS_PATH.name}: {len(labels)} clauses")
    for clause_type, count in sorted(by_type.items()):
        print(f"  {clause_type:16} {count}")


if __name__ == "__main__":
    build()
    build_hostile()
    build_mini()
