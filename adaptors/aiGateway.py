"""Model adapter - NOT YET IMPLEMENTED (provider undecided).

This is the only file in the project that knows which model is being used.
Everything else talks to it through one function, so swapping providers means
editing this file and the `model` line in config.yaml, nothing else.

CONTRACT - do not change without updating getSyntheticData.py:

    ask_ai(icl: str, questions: list[dict]) -> list[int]

    icl        the persona's full wave 1-3 text (~24k tokens), used as the
               in-context conditioning for this simulated respondent.

    questions  every question in ONE block, in order. Each dict has:
                   qid            str
                   row_id         int
                   question_type  "Matrix" | "MC" | "Slider" | "TE"
                   questiontext   str   already includes the matrix row label
                   options        list[str]   empty for Slider and TE
                   range_min      float
                   range_max      float

    returns    one int per question, SAME LENGTH and SAME ORDER as `questions`.
               Each int must fall in [range_min, range_max] for its question.
               For Matrix/MC that is a 1-based index into `options`.
               getSyntheticData.py clamps out-of-range values and raises if the
               length does not match.

PSEUDOCODE:

    def ask_ai(icl, questions):
        # 1. system prompt
        #    Frame as survey simulation, NOT identity role-play - phrasing like
        #    "You are simulating a survey respondent whose prior responses
        #    follow" refuses far less often than "You are a 38-year-old
        #    conservative", especially on the political blocks.
        #
        # 2. put `icl` FIRST and mark it cacheable
        #    It is identical across all 16 blocks for a persona, so a prefix
        #    cache makes blocks 2-16 cost ~10% of block 1. Anything that varies
        #    (the questions) must come AFTER the cache breakpoint or the cache
        #    never hits.
        #
        # 3. render the questions
        #    for i, q in enumerate(questions):
        #        "Q{i}: {q['questiontext']}"
        #        if q['options']: list them 1..n
        #        else:            state the valid range range_min..range_max
        #
        # 4. constrain the output with the provider's structured-output feature
        #    rather than asking politely in the prompt. Target schema:
        #        {"answers": [int, int, ...]}   length == len(questions)
        #    No prose, no reasoning, no floats, no nulls.
        #
        # 5. call the model, parse, return the list of ints
        #
        # 6. let exceptions propagate - getSyntheticData.py catches them,
        #    marks the block "pending", and a rerun retries only those.

Until this is implemented, getSyntheticData.py will mark every block pending.
"""


def ask_ai(icl: str, questions: list[dict]) -> list[int]:
    raise NotImplementedError(
        "aiGateway.ask_ai is not implemented yet - the model has not been chosen. "
        "See the contract and pseudocode in this file's docstring."
    )
