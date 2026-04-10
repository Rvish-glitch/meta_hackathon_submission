# Real-World Agent Eval — OpenEnv Environment

An [OpenEnv](https://openenv.dev)-compliant benchmark environment for evaluating AI agents on practical real-world tasks.

## Tasks

| Task | Difficulty | Steps | Score Range |
|---|---|---|---|
| Email Triage | Easy | 1 | −1.0 → 1.0 |
| Data Cleaning | Medium | ≤10 | 0.0 → 1.0 |
| Code Review | Hard | ≤15 | −1.0 → 1.0 |

### Email Triage
Classify 10 emails by **priority** (`urgent`/`normal`/`low`) and **category** (`action_required`/`fyi`/`spam`/`newsletter`). Special −0.10 penalty for marking an urgent email as spam.

### Data Cleaning
Fix a seeded CSV with: duplicate rows, null values, `$`-prefixed salary, float ages, inconsistent gender casing, and out-of-range values. Submit operations one at a time; graded on 8 deterministic checks.

### Code Review
Review a Python PR diff containing 5 seeded bugs:
- **B1** SQL injection via f-string interpolation
- **B2** Off-by-one in pagination
- **B3** Mutable default argument
- **B4** Missing `@require_auth` decorator on admin endpoint
- **B5** Division by zero

+0.15 per identified bug, +0.05 bonus if fix description contains the right keyword. −0.05 for false positives.

## Project Structure

```
├── environment/
│   ├── __init__.py
│   ├── env.py                  # OpenEnvEnvironment class + Pydantic models
│   ├── tasks/
│   │   ├── email_triage.py
│   │   ├── data_cleaning.py
│   │   └── code_review.py
│   └── graders/
│       ├── email_triage_grader.py
│       ├── data_cleaning_grader.py
│       └── code_review_grader.py
├── openenv.yaml
├── inference.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
HF_TOKEN=hf_xxx python inference.py
```

Optional overrides:
```bash
HF_TOKEN=hf_xxx \
HF_BASE_URL=https://api-inference.huggingface.co/v1 \
MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
python inference.py
```

## Use in Code

```python
from environment import OpenEnvEnvironment, Action, TaskName

# Single-step task
env = OpenEnvEnvironment("email_triage")
obs = env.reset()
print(obs.content)

action = Action(
    task=TaskName.EMAIL_TRIAGE,
    payload={
        "classifications": [
            {"email_id": "e01", "priority": "urgent", "category": "action_required"},
            # ...
        ]
    }
)
result = env.step(action)
print(result.reward.value)   # normalised score in [-1, 1]
print(result.reward.message) # human-readable breakdown

# Multi-step task
env = OpenEnvEnvironment("data_cleaning")
obs = env.reset()
result = env.step(Action(task="data_cleaning", payload={"operation": "drop_duplicates"}))
# ... more steps ...
result = env.step(Action(task="data_cleaning", payload={"operation": "submit"}))
print(result.reward.value)
```

## Docker

```bash
docker build -t openenv-eval .
docker run -e HF_TOKEN=hf_xxx openenv-eval
```

## Reward Design

All rewards are normalised to `[−1, 1]` (or `[0, 1]` for data cleaning):

- **Intermediate rewards** are issued after each step so agents get signal throughout multi-step tasks.
- **Graders are fully deterministic** — no LLM-as-judge. Scoring uses rule-based checks (keyword matching, schema validation, exact comparison).
- **Penalties** for clearly wrong actions (urgent→spam, false-positive bug reports) push the agent away from degenerate strategies.
