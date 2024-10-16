# It's a Smol World

This is a demo showing off the capabilities of the <SMOL MODEL NAME>

The demo is a command line interface representing a natural language interface to a mobile device making use of the <SMOL MODEL NAME> model.

## Setting up the environment

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

## Good Test Examples:

"I'd like to order 2 coffees from starbucks"

"I need a ride to SEATAC terminal A"

"What's the weather in san francisco today?"

"Text Remi and tell him the project is looking good"