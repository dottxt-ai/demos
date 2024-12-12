# .gifter

A Python tool that generates personalized gift ideas using structured generation with language models! Powered by holiday cheer and [.txt](https://dottxt.co)!

## Description

.gifter helps you find thoughtful gift ideas by analyzing descriptions of the recipient. It uses large language models to generate customized gift suggestions complete with descriptions, reasoning, and even card messages.

It optionally uses the [Exa API](https://exa.ai) to find additional information about recommendations! You'll need an API key from Exa to use this feature.

## Features

- Generate multiple gift ideas based on recipient descriptions
- Optional web UI for easy interaction, powered by Flask
- Each gift suggestion includes:
  - Gift type (e.g., book, electronics, experience)
  - Name and description
  - Reasoning for the suggestion
  - Personalized card message
  - Search query for finding similar items
- Optional search functionality to find real products
- Save gift ideas to JSON for later reference

## Installation

1. Clone this repository
```bash
git clone https://github.com/dottxt-ai/demos/
cd demos/holidays-2024/gifter
```
2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Add your Exa API key (`EXA_API_KEY`) to a `.env` file (optional, for search functionality)
```bash
echo "EXA_API_KEY=your_api_key" >> .env
```

## Usage

For simple command line usage, run:

```bash
python gifting.py
```

For the web UI, run:

```bash
python gift_web.py
```

## Example Output

```python
{
    "person_description": "My dad loves woodworking and jazz",
    "gift_reasoning": "Based on the recipient's interests in woodworking and jazz...",
    "gift_ideas": [
        {
            "gift_type": "tools",
            "name": "Premium Chisel Set",
            "description": "Professional-grade woodworking chisels...",
            "reason": "Perfect for detailed woodworking projects...",
            "card_message": "To help bring your woodworking visions to life...",
            "search_query": "professional woodworking chisel set reviews"
        },
        # ... more gift ideas ...
    ]
}
```

## Configuration

### Prompts

Prompts are found in `prompts/gifts.txt`. Modify these to change the behavior of the gift generator.

### Models

Models are defined at the beginning of `gifting.py`:

```python
# Models listed by increasing resource requirements
MODEL_OPTIONS = {
    "tiny": "HuggingFaceTB/SmolLM2-135M-Instruct",    # Minimal resources
    "small": "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # Balanced choice
    "medium": "NousResearch/Hermes-3-Llama-3.1-8B",    # Better quality
    "large": "meta-llama/Llama-3.3-70B-Instruct"       # Best quality
}

MODEL_STRING = MODEL_OPTIONS["tiny"]  # Default to balanced option
```

By default, the `tiny` model is used, as it can usually be run on a laptop.

If you have more resources available, you should experiment with larger models to increase recommendation quality.

### Web UI

Web templates are found in `templates/`, and can be modified as with any Flask app.

### Search

The search functionality uses the [Exa API](https://exa.ai). You'll need an API key from Exa to use this feature.

By default, .gifter uses Exa search with the following parameters:

```python
result = exa_client.search_and_contents(
    self.search_query,
    type="neural",
    use_autoprompt=True,
    num_results=3,
    highlights=True
)
```

You may wish to increase the number of results, or modify the search parameters.

## License

MIT

