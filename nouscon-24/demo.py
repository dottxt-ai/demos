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

# Set up the outlines image for modal. We use Python 3.11
# and install the outlines library and its dependencies.
modal_image = modal.Image\
    .debian_slim(python_version="3.11")\
    .pip_install(
        "outlines",
        "transformers",
        "accelerate",
        "sentencepiece",
        "bitsandbytes",
    )

# Let's specify the types of games we can choose from.
class GameSettingType(str, Enum):
    """
    The type of setting for the game. Game settings can be
    cyberpunk, solarpunk, or fantasy. The model will select
    one of these options.
    """
    cyberpunk = "cyberpunk"
    solarpunk = "solarpunk"
    fantasy = "fantasy"

class GameSetting(BaseModel):
    """
    A setting for the game. A GameSetting contains
    - A setting type (GameSettingType)
    - A description of the setting in natural language
    """
    setting: GameSettingType
    description: str

class CombatSkillLevel(str, Enum):
    """
    The skill level of the character in combat. Skills
    can be low, medium, or high, and correspond to a
    +1, +2, or +3 modifier to a die roll.
    """
    low = "low"
    medium = "medium"
    high = "high"

    def modifier(self):
        """
        Get the modifier for the skill level -- 
        skilllevel.modifier() will return 1, 2, or 3.
        """
        if self == CombatSkillLevel.low:
            return 1
        elif self == CombatSkillLevel.medium:
            return 2
        elif self == CombatSkillLevel.high:
            return 3

class Skills(BaseModel):
    """
    Skills for the character. This includes both
    attack and defense skills.
    """
    attack: CombatSkillLevel
    defense: CombatSkillLevel

    def attack_modifier(self):
        """
        Get the modifier for the attack skill.
        """
        return self.attack.modifier()

    def defense_modifier(self):
        """
        Get the modifier for the defense skill.
        """
        return self.defense.modifier()

# Here's a class for the characters.
class Character(BaseModel):
    """
    A character in the game. Characters have
    - A name
    - A description
    - Skills (which include attack and defense skill levels)
    - Health points (which start at 10 and go to 0)
    """
    name: str
    description: str
    skills: Skills
    health_points: int

    def __init__(self, **kwargs):
        """
        Initialize the character. This sets the health
        points to 10 and adds the attack and defense skills.

        The language model will set the health points to a 
        random number, so the init method overrides this to
        set the health points to 10.
        """
        super().__init__(**kwargs)
        # Reset health points to 10
        self.health_points = 10

    def attack(self, opponent):
        """
        Make an attack on the opponent. If the attacker rolls
        higher than the defender's defense skill, the attacker
        does the difference in rolls as damage to the defender.

        # Example:
        # Attacker rolls a 5
        # Defender rolls a 3
        # Attacker does 2 damage to defender
        """
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
        """
        Take damage from an attack.
        """
        self.health_points -= damage
        if self.health_points < 0:
            self.health_points = 0

# A class that represents a turn in combat
class Turn(BaseModel):
    """
    A turn in combat, containing only the description of the turn.
    """
    description: str

# Story
class Story(BaseModel):
    """
    A story for the game. A story contains
    - A setting (GameSetting)
    - Two characters (Character)
    - A reason for the battle (why the characters are fighting)
    - A title for the story (a short title for the story)
    """
    setting: GameSetting
    characters: Annotated[list[Character], Len(2)]
    reason_for_battle: str
    title_of_story: str

# The final story
class FinalStory(BaseModel):
    """
    The final story of the battle.

    - end_of_battle_description: a dramatic and concise description of 
      the final moments of the battle
    - implications_of_battle: a short description of the implications of the battle.
      What does this mean for the characters and the world?
    """
    end_of_battle_description: str
    implications_of_battle: str

# Modal specification. Here we choose an H100 GPU and specify
# the image we built above that has the outlines library installed.
@app.cls(gpu="H100", image=modal_image)
class Model:
    """
    Modal's build method is only called once when the container
    is built. We use this to download the model so we don't have 
    to do it on every request.
    """
    @modal.build()
    def download_model(self):
        import outlines

        # This downloads the model and caches it.
        outlines.models.transformers(
            llm,
        )

    @modal.enter()
    def setup(self):
        """
        This is called every time the container is entered. 
        We use this to load the model into the GPU so we don't
        have to do it on every request.
        """
        import outlines

        # The model is now stored in self.model so that we can use it
        # in the other methods.
        self.model = outlines.models.transformers(
            llm,
            device="cuda",
        )

    @modal.method()
    def make_story(self, prompt: str):
        """
        Make a Story object.
        """
        generator = outlines.generate.json(
            self.model,
            Story,

            # Here we set the temperature to 0.7 so we can get some
            # variability in the stories we get back.
            sampler=outlines.samplers.multinomial(temperature=0.7),
        )

        # Calling the generator like this will return a Story object.
        return generator(prompt)

    @modal.method()
    def describe_turn(self, prompt: str):
        """
        Describe a turn in the battle. This is called
        repeatedly throughout the battle to describe each
        turn as it happens.
        """
        generator = outlines.generate.json(self.model, Turn)
        return generator(prompt)

    @modal.method()
    def describe_final_story(self, prompt: str):
        """
        Describe the final story of the battle. This is called
        once the battle is over to summarize the battle and its
        implications.
        """
        generator = outlines.generate.json(self.model, FinalStory)
        return generator(prompt)


# The outlines.prompt decorator is used to simplify prompt templating.
# This prompt is used to describe to the model how to make a story.
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

# This prompt is used to describe to the model how to describe
# a turn in the battle.
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
=
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

# This prompt is used to describe to the model how to describe
# the final story of the battle.
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

# Local entrypoint. When you call
# 
# modal run demo.py
# 
# it will print out the entire arc of the story, including
# the setting, the characters, the reason for the battle,
# the turns of the battle, and the final story.
@app.local_entrypoint()
def main():
    # Create the Model object so we can call its methods.
    model = Model()

    # Set up our story. Model() constructs the object that
    # contains the model and its methods.
    story = model.make_story.remote(story_prompt())

    # If you are doing this locally, you can use
    # 
    # outlines_model = outlines.models.transformers(llm)
    # story_generator = outlines.generate.json(outlines_model, Story)
    # story = story_generator(story_prompt())

    # Set up panel width. This is only for pretty
    # printing 🤩
    panel_width = 60

    # Start printing out some information about the story
    print(Panel.fit(
        story.setting.description + f"\n\nSetting type: {story.setting.setting.value}",
        title=story.title_of_story,
        width=panel_width,
    ))

    # Print out the characters in the story so we can see
    # their stats and who they are.
    for character in story.characters:
        print(Panel.fit(
            character.description +
                f"\n\nAttack Skill: {character.skills.attack}\n" +
                f"Defense Skill: {character.skills.defense}\n",
            title=character.name,
            width=panel_width,
        ))

    # The characters are going to start fighting soon,
    # but why? Let's print out the reason.
    print(Panel.fit(
        story.reason_for_battle,
        title="Start of the battle",
        width=panel_width,
    ))

    # Create a list to store turns, which are natural language
    # descriptions of the actions that happen in the battle.
    turns = []

    # Main combat loop continues while all characters
    # still have health points.
    while all(character.health_points > 0 for character in story.characters):
        # Each character gets a turn to attack the other.
        for character in story.characters:
            # Get the opponent of this character.
            opponent = next(c for c in story.characters if c != character)

            # Check how much damage is done. Damage is 0 if the 
            # attack is unsuccessful.
            damage = character.attack(opponent)
            opponent.take_damage(damage)

            # If the opponent is defeated, break out of the loop
            if opponent.health_points <= 0:
                winner = character
                loser = opponent
                break

            # Turn the past turns into a string. This describes
            # the fight up to this point -- who struck who, how
            # has the environment changed, etc.
            action_history = story.setting.description + \
                "\n" + story.reason_for_battle + \
                "\n".join([f"{turn.description}" for turn in turns])

            # This converts the history of the battle into a full prompt
            # for the language model. This prompt describes what the
            # model should return, and the information it needs to do so.
            prompt = action_prompt(
                character,
                opponent,
                action_history,
                damage,
            )

            # Give the prompt to the model and ask it to describe 
            # how a character attacks another character.
            action_description = model.describe_turn.remote(prompt)

            # Add the action to the list of turns
            turns.append(action_description)

            # Make some strings to describe the damage done.
            damage_string = f"\n\n{character.name} did {damage} damage to {opponent.name}, leaving them with {opponent.health_points} health." if damage > 0 else ""
            no_damage_string = f"\n\n{character.name} failed to attack {opponent.name}." if damage == 0 else ""

            # Print out the action description
            print(Panel.fit(
                action_description.description +
                damage_string + no_damage_string,
                title=f"Turn {len(turns)}",
                width=panel_width,
            ))

    # If we get here, the battle is over and one character has won.
    # Print out who won!
    print(Panel.fit(
        f"{winner.name} won the battle!",
        title="Winner",
        width=panel_width,
    ))

    # Put the history of the battle into a prompt
    # asking the language model to describe the final moments
    # of the fight.
    prompt = final_prompt(
        "\n".join([f"{turn}" for turn in turns]),
        winner,
        loser,
    )

    # Summarize the final story. This is where the language
    # model is actually called.
    final_story = model.describe_final_story.remote(prompt)

    # Print out the end of the fight -- how did one character
    # defeat the other?
    print(Panel.fit(
        final_story.end_of_battle_description,
        title="The final moments",
        width=panel_width,
    ))

    # Print out the implications of the battle. What does this
    # mean for the characters and the world?
    print(Panel.fit(
        final_story.implications_of_battle,
        title="Implications",
        width=panel_width,
    ))
