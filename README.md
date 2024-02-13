## Overview

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type |  |
| Blog | - |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator checks that an LLM-generated summary covers the list of topics present in the document.

### Intended use

This validator is only intended for summarization. 

### Resources required

- Foundation model access keys: Yes (depending on the selected foundation model)

## Installation

```bash
$ guardrails hub install hub://guardrails/saliency_check
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails.hub import SaliencyCheck
from guardrails import Guard

# Initialize Validator
val = SaliencyCheck(
    docs_dir="/path/to/docs/dir",
    llm_callable="openai",
    on_fail="fix"
)

# Setup Guard
guard = Guard.from_string(
    validators=[val, ...],
)

guard.parse("LLM generated summary")
```

### Validating JSON output via Python

In this example, we apply the validator on the string field of a JSON output.

```python
# Import Guard and Validator
from pydantic import BaseModel
from guardrails.hub import SaliencyCheck
from guardrails import Guard

# Initialize Validator
val = SaliencyCheck(
    docs_dir="/path/to/docs/dir",
    llm_callable="openai",
    on_fail="fix"
)

# Create Pydantic BaseModel
class LLMOuput(BaseModel):
    summary: str = Field(
        description="LLM Generated Summary", validators=[val]
    )

# Create a Guard to check for valid Pydantic output
guard = Guard.from_pydantic(output_class=LLMOuput)

# Run LLM output generating JSON through guard
guard.parse("""
{
   "summary": "LLM generated summary"
}
""")
```

## API Reference

`__init__`

- `docs_dir`: Path to the directory containing the documents.
- `llm_callable`: Function that calls an LLM with a prompt, or `openai`
- `threshold`: Threshold for overlap between topics in document and summary.
- `on_fail`: The policy to enact when a validator fails.
