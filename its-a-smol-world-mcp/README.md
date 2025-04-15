# Outlines MCP Demo

This is a small update to the [It's a Smol World](https://github.com/dottxt-ai/demos/tree/main/its-a-smol-world) demo, adding Model Context Protocol (MCP) connectivity. 

The core concept remains the same: using a small language model for function calling, but now the client can connect to any MCP-compatible server instead of just using local functions. This means you can leverage the efficiency of a small local model for routing while accessing powerful external tools through the MCP protocol.

## Installation

### Windows

```bash
uv venv --python 3.11
.venv\Scripts\activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
```

## Usage

```bash
python .\src\app.py mcp-server\server.py -d
```

## Test Examples

- "Add 5 and 7"
- "I'd like to order two coffees from starbucks"
- "I need a ride to SEATAC terminal A"
- "What's the weather in san francisco today?"
- "Text Remi and tell him the project is looking good"