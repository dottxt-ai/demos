# Santa's Gift Delivery Game 🎅

A simple text-based game where you play as Santa Claus, delivering gifts to houses across a grid-based board.

But more importantly, it's a game where you can watch a language model attempt to play Santa.

The lanugage model uses Outlines and structured generation to determine a set of valid moves that the model must choose from.

![Gameplay](./santa-demo.gif)

We use [HuggingFaceTB/SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) as our language model, which is small enough for most users to run locally. It doesn't have the best performance, but it's a good starting point.

We'd love to see people attempt to make the language model play better! Please open a PR if you do.

## Description

In this game, you or a language model controls Santa as he moves around a 5x5 grid, collecting gifts (🎁) and delivering them to houses (🏠). The goal is to score points by successfully delivering gifts to houses.

## How to Play

- Santa (🎅) can move in 8 directions:
  - up, down, left, right, and diagonals
- Move to a gift (🎁) to pick it up
- Move to a house (🏠) while carrying a gift to deliver it
- Score 1 point for each successful delivery
- Type `x` to quit the game if you're playing manually, or Ctrl+C to quit the game if you're watching the language model play

## Game Rules

- You can only deliver gifts if you have them in your inventory
- Moving to a house without gifts won't score points
- New gifts spawn automatically to maintain 3 gifts on the board
- Houses remain in fixed positions throughout the game

## Requirements

- Python 3+
- Required packages:
  - outlines
  - torch
  - transformers
  - accelerate

## Installation

```bash
pip install outlines torch transformers accelerate
```

## Running the Game

```bash
python santa-game.py
```

## Game Modes

The game supports two modes:
- AI mode (default): The language model chooses moves
- Random mode: Moves are chosen randomly (set `random_mode=True` when running)
- Keyboard mode: You can control the game using the keyboard

## Modifications

- Modify the prompts in `prompts/` to change the model's behavior. In general, we recommend separating your prompts into folders for different types of games.
- Change the model in `santa-game.py` for possibly better-performing models.
- Modify game parameters in `santa-game.py`:
  - `board_size`: Change grid dimensions (default: 5x5)
  - `gift_count`: Number of gifts on board (default: 3)
  - `house_count`: Number of houses on board (default: 3)
  - `santa_char`: Character for Santa (default: 🎅)
  - `gift_char`: Character for gifts (default: 🎁)
  - `house_char`: Character for houses (default: 🏠)
  - `max_history`: Number of moves to store in history (default: 10)
    
## License

This project is open-sourced under the MIT License - see the LICENSE file for details.
