"""Model adapter - OpenAI Responses API.

This is the only file in the project that knows which model is being used.
Everything else talks to it through one function, so swapping providers means
editing this file and the `model` line in config.yaml, nothing else.

The API key is read from OPENAI_API_KEY in the `.env` file at the project root.

CONTRACT - do not change without updating getSyntheticData.py:

    ask_ai(icl: str, questions: list[dict], pid: int) -> list[int]

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

    pid        the persona id, used as the prompt cache key.

    returns    one int per question, SAME LENGTH and SAME ORDER as `questions`.
               For Matrix/MC that is a 1-based index into `options`.
               getSyntheticData.py clamps out-of-range values.

Exceptions propagate on purpose - getSyntheticData.py catches them, marks the
block "pending", and a rerun retries only those.
"""
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.common import CFG  # noqa: E402

load_dotenv(ROOT / ".env")
client = OpenAI(max_retries=CFG["max_retries"])  # reads OPENAI_API_KEY


class TokenRateLimiter:
    """Keeps the tokens sent in any 60 s window under the account's TPM limit.

    Cache reads count toward TPM too, so every request costs the full ~27k
    persona tokens. Without pacing, 8 threads fill the minute in a burst and
    the rest of the minute's requests fail with 429; with it they just wait.
    """

    def __init__(self, tokens_per_minute: int):
        self.cap = tokens_per_minute
        self.sent = deque()            # (time sent, tokens)
        self.lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.sent and now - self.sent[0][0] >= 60:
                    self.sent.popleft()
                if sum(t for _, t in self.sent) + tokens <= self.cap:
                    self.sent.append((now, tokens))
                    return
                wait = 60 - (now - self.sent[0][0])
            time.sleep(wait)


# a little under the real limit, since the token count below is an estimate
limiter = TokenRateLimiter(CFG["tpm_limit"])
CHARS_PER_TOKEN = 3.3   # measured ~3.5 on the persona text; lower = safer estimate

# Framed as survey simulation rather than identity role-play, which refuses
# less often on the political blocks.
SYSTEM_PROMPT = (
    "You are simulating a survey respondent for a research study. Below is the "
    "complete record of how this specific person answered an earlier set of "
    "surveys. Answer the new questions exactly as this person would, staying "
    "consistent with their demographics, attitudes, and the choices they made "
    "before. Do not answer as an average person or as an AI assistant.\n\n"
    "For a question with numbered options, answer with the option number. "
    "For a question with a numeric range, answer with a whole number inside "
    "that range."
)

# {"answers": [int, int, ...]} and nothing else
ANSWER_FORMAT = {
    "type": "json_schema",
    "name": "survey_answers",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answers": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["answers"],
        "additionalProperties": False,
    },
}


def render_questions(questions: list[dict]) -> str:
    """Number the questions and spell out the valid answers for each."""
    parts = []
    for i, q in enumerate(questions, 1):
        lines = [f"Q{i}: {q['questiontext']}"]
        if q["options"]:
            lines += [f"  {j}. {opt}" for j, opt in enumerate(q["options"], 1)]
            lines.append(f"  (answer with a number from 1 to {len(q['options'])})")
        else:
            lo, hi = int(q["range_min"]), int(q["range_max"])
            lines.append(f"  (answer with a whole number from {lo} to {hi})")
        parts.append("\n".join(lines))
    return (
        f"Answer all {len(questions)} questions below, in order, "
        f"as a list of exactly {len(questions)} integers.\n\n"
        + "\n\n".join(parts)
    )


def parse_answers(output_text: str, n: int) -> list[int]:
    """Pull the integer list out of the model's JSON, checking its length."""
    answers = json.loads(output_text)["answers"]
    if len(answers) != n:
        raise ValueError(f"model returned {len(answers)} answers for {n} questions")
    return answers


def ask_ai(icl: str, questions: list[dict], pid: int) -> list[int]:
    # Prompt caching (gpt-5.6+) works at breakpoints. The automatic breakpoint
    # lands at the end of the prompt, after the questions, so it never matched
    # the next block. Instead we mark the end of the icl explicitly: instructions
    # + icl are identical for all 16 blocks of a persona, so block 1 writes the
    # cache and blocks 2-16 read it. mode "explicit" stops the useless automatic
    # write. prompt_cache_key routes a persona's requests to the same cache.
    prompt_chars = len(SYSTEM_PROMPT) + len(icl) + len(render_questions(questions))
    limiter.acquire(int(prompt_chars / CHARS_PER_TOKEN))
    response = client.responses.create(
        model=CFG["model"],
        instructions=SYSTEM_PROMPT,
        input=[
            {"role": "user", "content": [{
                "type": "input_text",
                "text": f"PRIOR SURVEY RESPONSES:\n\n{icl}",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }]},
            {"role": "user", "content": render_questions(questions)},
        ],
        text={"format": ANSWER_FORMAT},
        prompt_cache_key=f"persona-{pid}",
        prompt_cache_options={"mode": "explicit"},
    )
    return parse_answers(response.output_text, len(questions))
