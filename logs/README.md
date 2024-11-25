# 😰 STRESSED - A Security Log Analysis System

**STRuctured Generation Security System Evaluating Data**

*"Everything is fine! This is fine! We're all fine!"*

## About

STRESSED is your friendly (if somewhat anxious) AI security intern, powered by Outlines for structured generation and fueled by virtual coffee.

STRESSED is a proof-of-concept that iterates through arbitrary logs in chunks, and for each chunk it will generate a summary of the logs and a list of potential security issues. 

STRESSED heavily applies structured generation techniques to review logs, using [Outlines](https://github.com/dottxt-ai/outlines).

Its intended use is to run on security logs from web servers, databases, etc. to help administrators understand what is happening in their systems at a high level.

A sample dataset of nginx web server logs is included for testing purposes, sourced from this [Kaggle dataset](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs). Apache and Linux logs are from [Loghub](https://github.com/logpai/loghub). Find samples in the `logs` directory.

## Features

- Produces strongly-typed JSON output using [Outlines](https://github.com/dottxt-ai/outlines).
- Analyzes web server logs with varying degrees of confidence
- Identifies security threats (or things it *thinks* might be threats)
- Generates detailed reports (with occasional stress-induced typos)
- Never takes vacation days or sick leave

## Example printout

STRESSED can print out the analysis in a human-readable format.

The report has several sections:

- Summary, a natural language summary of what occured in the chunk of logs reviewed.
- Key observations, a list of things that STRESSED notices. This resembles a chain-of-thought type process.
- Security events, a list of potential security events. This is followed by a list of log entries related to the possible event.
- Traffic patterns, a list of URLs and how they are being accessed. This is used to understand common access patterns for web servers.

## Installation

### Clone the repo

```bash
git clone https://github.com/dottxt-ai/stressed.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick start

```python
import outlines
import torch
from transformers import AutoTokenizer
from stressed import STRESSED  # Our anxious little helper

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
log_type = "web server"
prompt_template_path = "security-prompt.txt"

# Load the model
model = outlines.models.vllm(
    model_name,
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    disable_sliding_window=True,
    max_model_len=20000,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize our anxious intern!
parser = STRESSED(
    model=model,
    tokenizer=tokenizer,
    log_type=log_type,
    prompt_template_path=prompt_template_path,
    token_max=20000 # Maximum tokens to generate
)

# Choose the access log for giggles
log_path = "logs/access-10k.log"

# Load the logs into memory
with open(log_path, "r") as file:
    logs = file.readlines()

# Start the analysis
results = parser.analyze_logs(
    logs,
    chunk_size=20,
    format_output=True
)

# You can do stuff with the results here. results is a list of LogAnalysis objects.
for analysis in results:
    print(analysis.summary)
```

## Configuration

### Schema

By default, STRESSED uses the `LogAnalysis` schema. This schema was originally designed and
tested on nginx/apache web server logs, but could be used on other types of logs with minor
adjustments.

Any alternative schema must be a subclass of `BaseModel`, which [Outlines](https://github.com/dottxt-ai/outlines) uses to constrain the model to the given schema.


```python
# A LogAnalysis is a high-level analysis of a set of logs.
class LogAnalysis(BaseModel):
    summary: str
    observations: list[str]
    planning: list[str]
    events: list[WebSecurityEvent]
    traffic_patterns: list[WebTrafficPattern]
    highest_severity: Optional[SeverityLevel]
    requires_immediate_attention: bool
```

### Prompt template

A default security prompt template is provided as `security-prompt.txt`, but you can use your own by passing the `prompt_template_path` argument to the `STRESSED` constructor.

You __MUST__ include the following in your prompt template:

- `{log_type}`: The type of log you are parsing. This is typically something like "web server" or "database" so the model knows what kind of logs you are parsing.
- `{model_schema}`: The JSON schema for the output you expect. This is used to help the model understand the format of your output.
- `{logs}`: The logs you want to parse.

```
You are an expert security analyst reviewing {log_type} web server logs.

. . .

You should return valid JSON in the schema
{model_schema}

Here are the logs you need to parse:
{logs}

```


## Advanced Usage

### Customizing the Model

STRESSED supports any LLM that works with the Outlines library. You can customize the model settings when initializing:

```python
model = outlines.models.vllm(
    model_name,
    dtype=torch.bfloat16,  # Use different precision
    max_model_len=32000,   # Increase context length
    enable_prefix_caching=False,  # Disable caching
    # ... other vLLM parameters
)
```

### Chunk Size Considerations

The `chunk_size` parameter in `analyze_logs()` controls how many log entries are analyzed at once:

- Smaller chunks (10-20 lines) generally produce more detailed analysis
- Larger chunks process faster but may miss details
- Memory usage increases with chunk size
- The optimal size depends on your log complexity and model capabilities

### Output Formats

STRESSED provides two ways to handle results:

1. Structured objects: Each analysis returns a `LogAnalysis` object containing:
   - Summary text
   - Security events
   - Traffic patterns
   - Severity assessments
   - Observations and planning notes

2. Formatted console output (when `format_output=True`):
   - Rich text formatting with tables and colors
   - Organized sections for different analysis aspects
   - Highlighted security events and patterns

## Limitations

- Analysis quality depends on the underlying LLM
- May produce false positives/negatives
- Processing large log files requires chunking
- Model context length limits total logs per analysis
- Currently optimized for web server logs, may need adjustments for other log types

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Reduce chunk_size
   - Lower `max_model_len`
   - Use a smaller model

2. **Slow Processing**
   - Increase `chunk_size`
   - Enable prefix caching (enabled by default)
   - Use bfloat16 precision

3. **Poor Analysis Quality**
   - Try a different model
   - Reduce `chunk_size`
   - Adjust the prompt template
   - Ensure logs match expected format

## License

STRESSED is licensed under the MIT license. See the LICENSE file for details.

## Contributing

We welcome contributions to improve STRESSED's accuracy, reliability, and overall security posture. STRESSED may become nervous about merge conflicts.

Made with love by the [.txt](https://dottxt.co) team.
