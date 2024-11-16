# 😰 STRESSED - A Security Log Analysis System

**STRuctured Generation Security System Evaluating Data**

*"Everything is fine! This is fine! We're all fine!"*

## About

STRESSED is your friendly (if somewhat anxious) AI security intern, powered by Outlines for structured generation and fueled by virtual coffee.

STRESSED is a proof-of-concept that iterates through arbitrary logs in chunks, and for each chunk it will generate a summary of the logs and a list of potential security issues.

Its intended use is to run on security logs from web servers, databases, etc. to help administrators understand what is happening in their systems at a high level.

## Features

- Produces strongly-typed JSON output
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

```
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Log Analysis Report                                                                                                                        │
│ 2024-11-15 16:32:38                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

📝 Analysis Summary:
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Summary:                                                                                                                                   │
│ The logs show a high volume of GET requests to various product and image URLs, with some unusual POST requests. The traffic is primarily   │
│ from search engine bots and a few known user agents. No obvious security threats are detected.                                             │
│                                                                                                                                            │
│ Highest Severity: MEDIUM                                                                                                                   │
│ Requires Immediate Attention: True                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key Observations                                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ High volume of GET requests to product and image URLs. │
├────────────────────────────────────────────────────────┤
│ Unusual POST request to /ajaxFilter/p65,b1?page=2.     │
├────────────────────────────────────────────────────────┤
│ Traffic from search engine bots and known user agents. │
├────────────────────────────────────────────────────────┤
│ No unusual response codes or error messages.           │
└────────────────────────────────────────────────────────┘

⚠️  Security Events:
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Security Events     ┃ Details                                ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Unusual HTTP Method │ Type: Unusual HTTP Method              │
│                     │ Severity: MEDIUM                       │
│                     │ Confidence: 80.0%                      │
│                     │ Source IPs: 2.179.141.98               │
│                     │ URL Pattern: /ajaxFilter/p65,b1?page=2 │
│                     │ Possible Attacks: UNKNOWN              │
└─────────────────────┴────────────────────────────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Related Log Entries                                                                                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ LOGID-AK 2.179.141.98 - - [22/Jan/2019:03:56:40 +0330] "POST /ajaxFilter/p65,b1?page=2 HTTP/1.1" 200 │
│ 5092 "https://www.zanbil.ir/filter/p65,b1?page=1" "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36   │
│ (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36" "-"                                           │
│                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

📊 Traffic Patterns:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━┓
┃ URL Path                                                                                                    ┃ Method ┃ Hits ┃ Status Codes ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━┩
│ /product/14496/25676/%D8%B3%D8%B1-%D8%B4%D9%88%D8%B1-%D8%A2%D8%B1%DB%8C%D8%A7-%D8%B5%D9%86%D8%B9%D8%AA-%D9… │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /blog/tag/%DA%AF%D8%AC%D8%AA/                                                                               │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /product/21891/46412/%D8%B5%D9%86%D8%AF%D9%84%DB%8C-%D9%85%D8%A7%D8%B3%D8%A7%DA%98%D9%88%D8%B1-rain-sport-… │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /image/62178/productModel/150x150                                                                           │ GET    │ 2    │ 200: 2       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /filter/b481%2Cb874%2Cb226%2Cb570%2Cb598%2Cstexists%2Cb880%2Cb270%2Cb883%2Cb99%2Cb261%2Cb249%2Cb20%2Cb701%… │ GET    │ 2    │ 200: 2       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /static/images/amp/third-party/footer-mobile.png                                                            │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /product/30972/84309                                                                                        │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /filter/b656%2Cb703%2Cb67%2Cb226%2Cb41%2Cb598%2Cb168%2Cb723%2Cb597%2Cb88%2Cb548%2Cb6%2Cb679%2Cb215%2Cb105%… │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /image/34187?name=m12a-1.jpg&wh=200x200                                                                     │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /static/js/accordion.js?_=1548117042983                                                                     │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /filter/p5935%2Cb543                                                                                        │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /m/browse/refrigerator-and-freezer/%DB%8C%D8%AE%DA%86%D8%A7%D9%84-%D9%81%D8%B1%DB%8C%D8%B2%D8%B1            │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /product/21766?model=46248                                                                                  │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /image/59567/productModel/150x150                                                                           │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /blog/tag/%D8%A7%D8%AF%D9%88%DB%8C%D9%87/feed/                                                              │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /settings/logo                                                                                              │ GET    │ 1    │ 200: 1       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┼──────┼──────────────┤
│ /filter/p44%2Cv1%7C%D8%A2%D9%84%D8%A8%D8%A7%D9%84%D9%88%DB%8C%DB%8C%2Cv1%7C%D8%AE%D8%A7%DA%A9%D8%B3%D8%AA%… │ GET    │ 1    │ 200: 1       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┴──────┴──────────────┘
```

## Installation

### Clone the repo

```bash
git clone https://github.com/outlines-ai/stressed.git
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
from security_log_parser import STRESSED

# Choose a model and describe the type of logs you are parsing
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
log_type = "web server"
prompt_template_path = "security-prompt.txt"

# Load the model into memory
model = outlines.models.vllm(
    model_name,
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    disable_sliding_window=True,
    max_model_len=20000,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize our anxious intern
parser = STRESSED(
    model=model,
    tokenizer=tokenizer,
    log_type=log_type,
    prompt_template_path=prompt_template_path,
    token_max=20000
)

# Load the logs you want to parse
with open("archive/access.log", "r") as file:
    logs = file.readlines()[:1000]

results = parser.analyze_logs(
    logs,
    chunk_size=20,
    format_output=True
)

# Process results (hopefully nothing too alarming)
for analysis in results:
    # Process analysis results
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

## Contributing

We welcome contributions to improve STRESSED's accuracy, reliability, and overall security posture. STRESSED may become nervous about merge conflicts.

Made with love by the [.txt](https://dottxt.co) team.
