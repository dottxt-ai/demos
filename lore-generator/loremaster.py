from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

# ███████╗████████╗ ██████╗ ██████╗ ██╗   ██╗
# ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝
# ███████╗   ██║   ██║   ██║██████╔╝ ╚████╔╝
# ╚════██║   ██║   ██║   ██║██╔══██╗  ╚██╔╝
# ███████║   ██║   ╚██████╔╝██║  ██║   ██║
# ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝
#
# ███████╗████████╗██████╗ ██╗   ██╗ ██████╗████████╗██╗   ██╗██████╗ ███████╗
# ██╔════╝╚══██╔══╝██╔══██╗██║   ██║██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝
# ███████╗   ██║   ██████╔╝██║   ██║██║        ██║   ██║   ██║██████╔╝█████╗
# ╚════██║   ██║   ██╔══██╗██║   ██║██║        ██║   ██║   ██║██╔══██╗██╔══╝
# ███████║   ██║   ██║  ██║╚██████╔╝╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝

class LoreEntry(BaseModel):
    """
    A class representing a lore entry in the database.
    Lore entries are descriptive pieces of information that
    add color to a world.
    """
    name: str
    content: str
    keywords: List[str]

    # Embed the content of the lore entry
    def encode(self, embedding_fn):
        return embedding_fn(self.content)

    # Insert the lore entry into the database
    def insert(self, client, embedding_fn):
        # Convert the lore entry to a dictionary and add the embedding
        dict_data = self.model_dump()
        dict_data["vector"] = self.encode(embedding_fn)

        # milvus it, bay-bee
        client.insert(
            collection_name="lore",
            data=[dict_data],
            fields=["content"],
        )

class SettingType(str, Enum):
    """
    The setting of the world. This is an Enum, so
    the language model must choose from one of the
    following options.
    """
    science_fiction = "science_fiction"
    fantasy = "fantasy"
    horror = "horror"
    cyberpunk = "cyberpunk"
    steampunk = "steampunk"
    post_apocalyptic = "post_apocalyptic"
    magical_realism = "magical_realism"

class World(BaseModel):
    setting: SettingType
    world_description: str

    def to_text(self):
        return f"""Genre: {self.setting}\nWorld Description: {self.world_description}"""

    def world_proposal_prompt(world_type=None):
        """
        This function returns the system prompt for the world proposal prompt.

        Returns:
            (system_prompt, user_prompt)
        """

        # System prompt is first, user prompt is second
        return (f"""
        You are a world builder. Your job is to
        describe a brand new world with a setting and a description of the world.

        {"The user has offered this guidance to you: " + world_type if world_type else ""}

        The setting may be one of the following:
        - fantasy
        - science fiction
        - horror
        - cyberpunk
        - steampunk
        - post-apocalyptic
        - magical realism

        The world description should be comprehensive. It should describe the unique features of the world,
        focusing primarily on physical properties, geography, and culture.

        Be extremely brief. We'll add more information later.
        """,
        # User prompt
        """
        Please provide a world description and a setting. This should be the seed from which
        the rest of the lore of the world is built.

        Be brief.

        Please provide your world description and setting in a JSON format.
        """
        )

    def event_proposal_prompt(self):
        system_prompt = """
        You propose entries for a lore database. These should be general plot points,
        characters, locations, etc. Focus on the abstraction like "a hero rises" or
        "the location of a battle".

        Lore entries are pieces of information that add color to a world.
        They can be anything, including:
        - a historical event
        - a description of a character
        - a description of a location
        - a description of an object
        - a description of a concept

        You have access to all the lore of the world. To access this lore, you must
        provide a list of requests that will be used to further refine the proposal.
        Queries you provide should clarify characters involved, institutions that
        exist, past events, etc.

        Queries are used to make sure that the entries you propose are consistent with the lore
        of the world. They should try to understand whether the entry is plausible given
        the existing lore. Your queries should focus on existing lore --
        are there relevant characters? Conflicts? Locations?
        """

        user_prompt = f"""
        # World Summary

        {self.to_text()}

        Please provide:
        - A proposal for a lore entry in natural language.
        - A list of information requests that will be used to refine the proposal.
          These should be in the form of natural language queries like "Does a
          character with these qualities exist?" or other general questions to
          understand the world.

        Be brief.

        Please provide your proposal and information requests in a JSON format.
        """

        return system_prompt, user_prompt

class LoreEntryCandidate(BaseModel):
    proposal: str
    reasoning_steps: List[str]
    information_requests: List[str]

class InformationRequestAnswer(BaseModel):
    reasoning_steps: List[str] # this forces the model to provide reasoning
    answer: str

    def answer_prompt(proposal: str, query: str, search_results: List[LoreEntry]):
        system_prompt = """
        You are a world builder, designed to take a proposal for a lore entry and
        refine it based on information requests. An agent has already
        provided a proposal for a lore entry, a list of information
        requests, and a list of lore entries that are potentially
        relevant.

        You are reviewing the agent's question. You will be provided
        the question and a list of lore entries that are potentially
        relevant. Your job is to determine whether the question is
        asking for information that is already known in the lore. If it is,
        you should provide a concise answer to the agent's question.

        If the question does not seem to be answerable with the provided lore,
        you should tell the agent that the information is not yet known. Your
        job is to review and guide the proposal to be sure that it is consistent
        with the lore of the world.

        You may also provide a list of reasoning steps that lead from the available
        lore to the answer.
        """

        user_prompt = f"""
        Proposal being refined:

        {proposal}

        Information Request from agent:

        {query}

        Search results:

        {search_results}

        Be brief.

        Make sure your answer uses only information from the search results.
        Do not make up information.
        """

        return system_prompt, user_prompt

def prompt_refine_proposal(lore_query: str):
    return f"""
    <|im_start|>system
    You are a world builder, designed to take a proposal for a lore entry and
    refine it based on information requests. An agent has already
    provided a proposal for a lore entry, a list of information
    requests, and a list of lore entries that are potentially
    relevant.

    Your job is to produce a new fact that should be added to the lore of the world.
    It may be brief, like a sentence or two, or it may be longer, like a paragraph.
    It should be a fact that would be relevant to someone trying to learn the lore
    of the world.

    You may add as much or as little detail as you think is appropriate, as long as
    it is consistent with the lore of the world.

    Provide reasoning steps that lead to the new lore entry.

    <|im_end|>
    <|im_start|>user

    Here is the content you will need to refine the proposal:

    {lore_query}

    Be brief.

    <|im_end|>
    <|im_start|>assistant
    """

def separator():
    print("\n")
    print("─" * 60)
    print("\n")