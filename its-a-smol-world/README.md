# The Bunny B1: Powered by SmolLM2

This is a demo to celebrate the release of [the `SmolLM2-1.7B` model](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) from Hugging Face 🤗!

Ever want to have a natural language interface to local apps? The Bunny B1 demonstrates how to combine the power of SmolLM2 with structured generation using [Outlines](https://github.com/dottxt-ai/outlines) to be able to map natural language requests to calls to applications, even on smaller devices.

Here's a look at the demo in action:

![Bunny B1](./demo.gif)

## Setting up the environment

Before install the requirements, please be sure to have a Rust compiler installed, otherwise download it from the [official website](https://www.rust-lang.org/tools/install) using:  

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then you can install the required libraries using:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the demo

To start the demo, run the following command:

```bash
python3 ./src/app.py
```

The demo provides an interface for natural language interaction with a mobile device. You can provide natural language commands and the model will choose one of the following actions:

- Send a text message
- Order a food delivery
- Order a ride
- Get the weather

To add a new function you can edit `functions.json` and follow the pattern you'll find in the examples.

## Good Test Examples:

"I'd like to order two coffees from starbucks"

"I need a ride to SEATAC terminal A"

"What's the weather in san francisco today?"

"Text Remi and tell him the project is looking good"

## Customizing

The `constants.py` file allows you to customize the model, device, and torch tensor type. This demo was created on a Mac so the default device is `mps`. You can swap this out for `cuda` if you'd like.
