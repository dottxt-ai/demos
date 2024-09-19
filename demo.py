import random
import outlines
import modal

from pydantic import BaseModel, Field
from enum import Enum
from typing import Annotated
from annotated_types import Len

from rich import print
from rich.panel import Panel

# Make our app
app = modal.App("game-master")

# What language model will we use?
llm = "NousResearch/Hermes-3-Llama-3.1-8B"

# Set up the outlines image
modal_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "outlines",
    "transformers",
    "accelerate",
    "sentencepiece",
    "bitsandbytes",
    "vllm"
)

# Let's specify the types of games we can choose from.
class GameSettingType(str, Enum):
    cyberpunk = "cyberpunk"
    solarpunk = "solarpunk"
    fantasy = "fantasy"

class GameSetting(BaseModel):
    setting: GameSettingType
    description: str

class CombatSkillLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

    def modifier(self):
        if self == CombatSkillLevel.low:
            return 1
        elif self == CombatSkillLevel.medium:
            return 2
        elif self == CombatSkillLevel.high:
            return 3

class Skills(BaseModel):
    attack: CombatSkillLevel
    defense: CombatSkillLevel

    def attack_modifier(self):
        return self.attack.modifier()

    def defense_modifier(self):
        return self.defense.modifier()

# Here's a class for the characters.
class Character(BaseModel):
    name: str
    description: str
    skills: Skills
    health_points: int = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Reset health points to 10
        self.health_points = 10

    def attack(self, opponent):
        # Get the modifiers for your attack and your opponent's defense
        attack_modifier = self.skills.attack_modifier()
        defense_modifier = opponent.skills.defense_modifier()

        # Roll a 10-sided die and add the attack modifier
        attack_roll = random.randint(1, 10) + attack_modifier
        defense_roll = random.randint(1, 10) + defense_modifier

        # Calculate damage
        if attack_roll > defense_roll:
            return attack_roll - defense_roll
        else:
            return 0

    def take_damage(self, damage):
        self.health_points -= damage
        if self.health_points < 0:
            self.health_points = 0

# A class that represents a turn in combat
class Turn(BaseModel):
    description: str

# Story
class Story(BaseModel):
    setting: GameSetting
    characters: Annotated[list[Character], Len(2)]
    reason_for_battle: str
    title_of_story: str

# A turn in combat
class Turn(BaseModel):
    description: str

# The final story
class FinalStory(BaseModel):
    end_of_battle_description: str
    implications_of_battle: str

@app.cls(gpu="H100", image=modal_image)
class Model:
    @modal.build()
    def download_model(self):
        import outlines
        outlines.models.transformers(
            llm,
        )

    @modal.enter()
    def setup(self):
        import outlines
        self.model = outlines.models.transformers(
            llm,
            device="cuda",
        )

    @modal.method()
    def make_story(self, prompt: str):
        generator = outlines.generate.json(
            self.model,
            Story,
            sampler=outlines.samplers.multinomial(temperature=0.7),
        )
        return generator(prompt)

    @modal.method()
    def describe_turn(self, prompt: str):
        generator = outlines.generate.json(self.model, Turn)
        return generator(prompt)

    @modal.method()
    def describe_final_story(self, prompt: str):
        generator = outlines.generate.json(self.model, FinalStory)
        return generator(prompt)

@outlines.prompt
def story_prompt():
    """
    <|im_start|>user
    You are the best game master in the world. You are tasked with creating a story
    for a roleplaying game. The game features two characters fighting one another.

    Select a setting from the following options:

    - Cyberpunk
    - Solarpunk
    - Fantasy

    You'll need to create

    - A scenario, which includes a setting, characters, and a description of the encounter
    - A reason for the battle
    - A title for the story

    When describing the setting, speak in general terms -- how does the world look
    and feel? What are key features of the setting? What conflict exists in the
    broader setting?

    When you get to the reason for the battle, place the two characters in a situation
    where they are forced to fight one another that connects to the broader setting.

    Characters have skills, which may be low, medium, or high.

    - Attack Skill
    - Defense Skill

    The reason for the battle should be short, concise, and set up combat immediately.

    Return the story in JSON format.
    <|im_end|>
    <|im_start|>assistant
    """

@outlines.prompt
def action_prompt(
    attacker: Character,
    defender: Character,
    history: str,
    damage: int,
):
    """
    <|im_start|>user
    You are the best game master in the world. Your role is to take
    die rolls and character statistics to create a dramatic and engaging story
    by describing the action of the story as it unfolds.

    Here's what happened just now:

    Two characters are fighting one another.
    One character ({{attacker.name}}) has attacked the other character ({{defender.name}}).
    The attack was {{"successful" if damage > 0 else "unsuccessful"}}.

    The defender has {{defender.health_points}} health points remaining.
    {{"The defender has been defeated." if defender.health_points <= 0 else ""}}

    # BEGIN INPUT

    Attacker information:
    name: {{attacker.name}}
    description: {{attacker.description}}

    Defender information:
    name: {{defender.name}}
    description: {{defender.description}}

    History of actions:

    {{history}}\

    # YOUR TASK

    Describe {{attacker.name}}'s attack on {{defender.name}} in a single sentence.
    Try to make the description dramatic, engaging, and concise. Readers should feel
    like they are in the middle of the action.

    You should describe the attack in a way that makes sense for the setting.
    For example, if the setting is cyberpunk, the attack might be someone shooting
    a gun, and the defense might be someone ducking behind a car.

    If the setting is fantasy, the attack might be a spell or sword attack,
    and the defense might be a spell to teleport a few feet away, or to
    block with a shield.

    Characters will attack each other and succeed or fail based on their skills -- the defender
    may dodge/block/evade the attack if the attack is unsuccessful.

    In the event of an unsuccessful attack, no damage is done. Try to convey this
    with words. The defender should dodge, block, or evade the attack skillfully.

    Be concise!

    Your response:
    <|im_end|>
    <|im_start|>assistant
    """

@outlines.prompt
def final_prompt(
    history: str,
    winner: Character,
    loser: Character,
):
    """
    <|im_start|>user
    You are the best game master in the world. Your role is to take
    die rolls and character statistics to create a dramatic and engaging story
    by describing the action of the story as it unfolds.

    Two characters fought each other.
    One character ({{winner.name}}) has defeated the other character ({{loser.name}}).

    Provide two elements in JSON format:

    - end_of_battle_description: a dramatic and concise description of the final moments of the battle
    - implications_of_battle: a short description of the implications of the battle. What does this
    mean for the characters and the world?

    Winner information:
    name: {{winner.name}}
    description: {{winner.description}}
    skills: {{winner.skills.attack.value}} (attack) {{winner.skills.defense.value}} (defense)

    Loser information:
    name: {{loser.name}}
    description: {{loser.description}}
    skills: {{loser.skills.attack.value}} (attack) {{loser.skills.defense.value}} (defense)

    History of actions:

    {{history}}

    Your response:
    <|im_end|>
    <|im_start|>assistant
    """

# Local entrypoint
@app.local_entrypoint()
def main():
    # Set up our story
    story = Model().make_story.remote(story_prompt())

    # Set up panel width
    panel_width = 60

    # Start printing out some information about the story
    print(Panel.fit(
        story.setting.description + f"\n\nSetting type: {story.setting.setting.value}",
        title=story.title_of_story,
        width=panel_width,
    ))

    for character in story.characters:
        print(Panel.fit(
            character.description +
                f"\n\nAttack Skill: {character.skills.attack}\n" +
                f"Defense Skill: {character.skills.defense}\n",
            title=character.name,
            width=panel_width,
        ))

    print(Panel.fit(
        story.reason_for_battle,
        title="Start of the battle",
        width=panel_width,
    ))

    # character names
    c1_name = story.characters[0].name
    c2_name = story.characters[1].name

    # Create a list to store turns
    turns = []

    # Main combat loop
    while all(character.health_points > 0 for character in story.characters):
        for character in story.characters:
            # Get the opponent
            opponent = next(c for c in story.characters if c != character)

            # Check if the character's attack is successful
            damage = character.attack(opponent)
            opponent.take_damage(damage)

            # If the battle is over, break out of the loop
            if opponent.health_points <= 0:
                winner = character
                loser = opponent
                break

            # Turn the past turns into a string
            action_history = story.setting.description + \
                "\n" + story.reason_for_battle + \
                "\n".join([f"{turn.description}" for turn in turns])

            # Ask the model to describe the action
            prompt = action_prompt(
                character,
                opponent,
                action_history,
                damage,
            )

            action_description = Model().describe_turn.remote(prompt)

            # Add the action to the list of turns
            turns.append(action_description)

            # Print out the action description
            damage_string = f"\n\n{character.name} did {damage} damage to {opponent.name}, leaving them with {opponent.health_points} health." if damage > 0 else ""
            no_damage_string = f"\n\n{character.name} failed to attack {opponent.name}." if damage == 0 else ""
            print(Panel.fit(
                action_description.description +
                (damage_string if damage > 0 else no_damage_string),
                title=f"Turn {len(turns)}",
                width=panel_width,
            ))

    # If we get here, the battle is over and one character has won.
    # Print out the final health points
    # winner = next(c for c in story.characters if c.health_points > 0)
    # loser = next(c for c in story.characters if c.name != winner.name)

    print(Panel.fit(
        f"{winner.name} won the battle!",
        title="Winner",
        width=panel_width,
    ))

    # Put the history of the battle into a prompt
    # asking the language model to describe the final moments
    # of the battle.
    prompt = final_prompt(
        "\n".join([f"{turn}" for turn in turns]),
        winner,
        loser,
    )

    # Summarize the final story
    final_story = Model().describe_final_story.remote(prompt)

    print(Panel.fit(
        final_story.end_of_battle_description,
        title="The final moments",
        width=panel_width,
    ))

    print(Panel.fit(
        final_story.implications_of_battle,
        title="Implications",
        width=panel_width,
    ))



# Let's start by creating our setting.
# story_generator = outlines.generate.json(model, Story)

# @outlines.prompt
# def story_prompt():
#     """
#     You are the best game master in the world. You are tasked with creating a setting
#     for a roleplaying game. The game features two characters fighting one another.


#     """

#
# story = story_generator()