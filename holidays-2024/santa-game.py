import random
import os
import time
from typing import Tuple

import outlines
import torch
from transformers import AutoTokenizer
# model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
model_name = "Qwen/Qwen2.5-14B"
# model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# model_name = "microsoft/Phi-3-mini-128k-instruct"
model = outlines.models.transformers(
    model_name,
    device="auto",
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

def template(prompt):
    templated = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return templated

def get_valid_moves(santa_pos: list, board_size: int) -> str:
    """Returns a string of valid moves based on Santa's position"""
    moves = []
    if santa_pos[0] > 0:  # Can move up
        moves.append("up")
    if santa_pos[0] < board_size - 1:  # Can move down
        moves.append("down")
    if santa_pos[1] > 0:  # Can move left
        moves.append("left")
    if santa_pos[1] < board_size - 1:  # Can move right
        moves.append("right")
    if santa_pos[0] > 0 and santa_pos[1] > 0:  # Can move up-left
        moves.append("up-left")
    if santa_pos[0] > 0 and santa_pos[1] < board_size - 1:  # Can move up-right
        moves.append("up-right")
    if santa_pos[0] < board_size - 1 and santa_pos[1] > 0:  # Can move down-left
        moves.append("down-left")
    if santa_pos[0] < board_size - 1 and santa_pos[1] < board_size - 1:  # Can move down-right
        moves.append("down-right")

    # Randomly shuffle the moves
    random.shuffle(moves)
    return "|".join(moves)

class SantaGame:
    def __init__(self):
        self.board_size = 5
        self.santa_pos = [4, 4]  # Start Santa in middle
        self.score = 0
        self.gift_count = 3  # Number of gifts on board at once
        self.house_count = 3  # Number of houses on board
        self.santa_char = '🎅'
        self.gift_char = '🎁'
        self.house_char = '🏠'
        self.blank_char = '  '
        self.inventory = 0
        self.board = [[self.blank_char for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.output_buffer = ""
        self.move_history = []  # Add move history list
        self.max_history = 10    # Store last 5 moves
        self.action_log = []    # Add action log list

    def spawn_gifts(self):
        while sum(row.count(self.gift_char) for row in self.board) < self.gift_count:
            x, y = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
            if self.board[y][x] == self.blank_char:
                self.board[y][x] = self.gift_char

    def spawn_houses(self):
        while sum(row.count(self.house_char) for row in self.board) < self.house_count:
            x, y = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
            if self.board[y][x] == self.blank_char:
                self.board[y][x] = self.house_char

    def get_distances_description(self) -> str:
        """Returns a natural language description of distances to gifts and houses"""
        description = []
        
        # Find gifts and houses
        gifts = []
        houses = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == self.gift_char:
                    gifts.append((y, x))
                elif self.board[y][x] == self.house_char:
                    houses.append((y, x))
        
        # Describe gifts
        if gifts:
            gift_descriptions = []
            for y, x in gifts:
                vert = y - self.santa_pos[0]
                horiz = x - self.santa_pos[1]
                dirs = []
                if abs(vert) > 0:
                    dirs.append(f"{abs(vert)} step{'s' if abs(vert)>1 else ''} {'up' if vert<0 else 'down'}")
                if abs(horiz) > 0:
                    dirs.append(f"{abs(horiz)} step{'s' if abs(horiz)>1 else ''} {'left' if horiz<0 else 'right'}")
                gift_descriptions.append(" and ".join(dirs))
            description.append("Gifts are: " + "; ".join(gift_descriptions))
        
        # Describe houses
        if houses:
            house_descriptions = []
            for y, x in houses:
                vert = y - self.santa_pos[0]
                horiz = x - self.santa_pos[1]
                dirs = []
                if abs(vert) > 0:
                    dirs.append(f"{abs(vert)} step{'s' if abs(vert)>1 else ''} {'up' if vert<0 else 'down'}")
                if abs(horiz) > 0:
                    dirs.append(f"{abs(horiz)} step{'s' if abs(horiz)>1 else ''} {'left' if horiz<0 else 'right'}")
                house_descriptions.append(" and ".join(dirs))
            description.append("Houses are: " + "; ".join(house_descriptions))
        
        return "\n".join(description)

    def get_game_state(self) -> str:
        self.output_buffer = ""
        
        # Add game status
        self.output_buffer += f"Score: {self.score} | Gifts: {self.inventory}\n"
        
        # Add board with pipes and underscores
        for y in range(self.board_size):
            row = f"row {y} |"
            for x in range(self.board_size):
                if [y, x] == self.santa_pos:
                    row += self.santa_char
                else:
                    row += self.board[y][x]
            self.output_buffer += row + "\n"
        
        return self.output_buffer

    def move_santa(self, direction: str) -> bool:
        new_pos = self.santa_pos.copy()
        
        if direction == 'up' and new_pos[0] > 0:
            new_pos[0] -= 1
        elif direction == 'down' and new_pos[0] < self.board_size - 1:
            new_pos[0] += 1
        elif direction == 'left' and new_pos[1] > 0:
            new_pos[1] -= 1
        elif direction == 'right' and new_pos[1] < self.board_size - 1:
            new_pos[1] += 1
        elif direction == 'up-left' and new_pos[0] > 0 and new_pos[1] > 0:
            new_pos[0] -= 1
            new_pos[1] -= 1
        elif direction == 'up-right' and new_pos[0] > 0 and new_pos[1] < self.board_size - 1:
            new_pos[0] -= 1
            new_pos[1] += 1
        elif direction == 'down-left' and new_pos[0] < self.board_size - 1 and new_pos[1] > 0:
            new_pos[0] += 1
            new_pos[1] -= 1
        elif direction == 'down-right' and new_pos[0] < self.board_size - 1 and new_pos[1] < self.board_size - 1:
            new_pos[0] += 1
            new_pos[1] += 1
        else:
            return False

        if self.board[new_pos[0]][new_pos[1]] == self.gift_char:
            self.inventory += 1
            self.board[new_pos[0]][new_pos[1]] = self.blank_char
            self.action_log.append("You picked up a gift! 🎁")
        elif self.board[new_pos[0]][new_pos[1]] == self.house_char and self.inventory > 0:
            self.score += 1
            self.inventory -= 1
            self.action_log.append(f"You delivered {self.score} gifts! 🏠")
            
        self.santa_pos = new_pos
        return True

    def play_step(self, moves: list[str], clear_screen: bool = False) -> tuple[str, bool]:
        """
        Executes a single game step and returns the game state
        Returns: (game_state: str, is_game_over: bool)
        """
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        
        for move in moves:  # Process each move in the list
            if move == 'q':
                return f"Game Over! Final Score: {self.score}", True
                
            if move in ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right']:
                self.action_log.append(f"Move: {move}")
                self.move_santa(move)
                self.spawn_gifts()
                
                # Store current game state in history
                current_state = (
                    "\n".join(self.action_log[-3:]) +  # Show last 3 actions
                    "\n" + self.get_game_state()
                )
                self.move_history.append(current_state)
                if len(self.move_history) > self.max_history:
                    self.move_history.pop(0)
            
        return self.get_game_state(), False

    def play(self, random_mode=False, clear_screen: bool = False):
        """
        Interactive version of the game for terminal play
        Args:
            random_mode (bool): If True, moves are chosen randomly instead of using AI
            clear_screen (bool): If True, clears the terminal before each board render
        """
        self.spawn_houses()  # Spawn houses once at the start
        self.spawn_gifts()

        while True:
            if clear_screen:
                os.system('cls' if os.name == 'nt' else 'clear')
            print(self.get_game_state())
            
            valid_moves = get_valid_moves(self.santa_pos, self.board_size)
            moves = []
            
            if random_mode:
                # Split the valid_moves string into a list and choose randomly
                move = random.choice(valid_moves.split('|'))
                moves.append(move)
            else:
                choose_direction = outlines.generate.regex(
                    model,
                    f"({valid_moves})",
                    sampler=outlines.samplers.multinomial(top_k=3)
                )
                
                    # You are Santa, a kind and generous person who delivers gifts to
                    # the world. You are currently on a board of size {self.board_size}x{self.board_size}.
                    # You can move up, down, left, or right.

                    # Move Santa ({valid_moves}) or q to quit.

                    # Your goal is to pick gifts up (your current inventory is {self.inventory})
                    # and deliver them to the houses (your current score is {self.score}).

                    # Your general strategy is:
                    # - Move towards the nearest gift
                    # - If you are carrying a gift, move towards the nearest house
                    # - If you are not carrying a gift, move towards the nearest gift

                    # Helpful information:
                    # {self.get_distances_description()}
                prompt = f"""
                    # Instructions

                    You are Santa ({self.santa_char}). You pick up gifts and 
                    deliver them to houses. Move to a gift ({self.gift_char})
                    to pick it up, or to a house ({self.house_char}) to deliver
                    your gifts. You get 1 point for each gift delivered. 

                    Moving to a house when you have no gifts is a waste of time.

                    # Example

                    Board state:

                    Score: 0 | Gifts: 1
                    row 0 |  🎁      
                    row 1 |          
                    row 2 |🏠        
                    row 3 |    🎅    
                    row 4 |🎁🏠🎁  🏠
                    
                    Choice: down

                    New board state:

                    Score: 0 | Gifts: 2
                    row 0 |  🎁      
                    row 1 |          
                    row 2 |🏠      🎁 
                    row 3 |        
                    row 4 |🎁🏠🎅  🏠

                    # Current board

                    You currently have {self.inventory} gifts and a score of {self.score}.
                    
                    Current board:
                    {self.get_game_state()}

                    Your move, choose from ({valid_moves}) or q to quit:
                """
                move = choose_direction(template(prompt))

                # First character is the number, the rest is the move. 
                # Add n times the move to the moves list
                print(move)
                moves.append(move)

            # Update game state
            game_state, is_game_over = self.play_step(moves, clear_screen)


            if is_game_over:
                print(game_state)
                break

if __name__ == "__main__":
    game = SantaGame()
    game.play(random_mode=False, clear_screen=True)  # Set clear_screen to control screen clearing
