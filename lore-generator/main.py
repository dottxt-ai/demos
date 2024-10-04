from enum import Enum
import random
from typing import List

from pydantic import BaseModel
from pydantic.tools import parse_obj_as
from pymilvus import IndexType, MilvusClient, FieldSchema, DataType
from pymilvus import model #for embeddings

from rich import print
from rich.panel import Panel

# load the milvus client
client = MilvusClient("milvusdemo.db")

# drop the collection if it already exists,
# since this script bootstraps the databases
if client.has_collection(collection_name="lore"):
    client.drop_collection(collection_name="lore")

# create the collection. default distance metric is "COSINE"
client.create_collection(
    collection_name="lore",
    dimension=768,
    auto_id=True,
)

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

class LoreEntry(BaseModel):
    name: str
    content: str
    keywords: List[str]

    def encode(self, embedding_fn):
        return embedding_fn.encode_documents([self.content])[0]

    def insert(self, client, embedding_fn):
        dict_data = self.model_dump()
        dict_data["vector"] = self.encode(embedding_fn)

        print(f"Inserting {self.name}")
        print(f"Dimensions: {dict_data['vector'].shape}")
        client.insert(
            collection_name="lore",
            data=[dict_data],
            fields=["content"],
        )

class SettingType(str, Enum):
    fantasy = "fantasy"
    science_fiction = "science_fiction"

class World(BaseModel):
    setting: SettingType
    world_description: str

    def to_text(self):
        return f"""Genre: {self.setting}\nWorld Description: {self.world_description}"""

    def world_proposal_prompt():
        return f"""
        <|im_start|>system
        You are a world builder. You are given a world description and a setting. Your job is to
        describe a brand new world with a setting, a description of the world, and a list of characters.

        The setting must be either "fantasy" or "science_fiction".

        The world description should be comprehensive. It should describe the unique features of the world,
        focusing primarily on physical properties, geography, and culture.

        Be extremely brief. We'll add more information later.

        <|im_end|>
        <|im_start|>user

        Please provide a world description, a setting, and a list of characters.
        <|im_end|>
        <|im_start|>assistant
        """

    def event_proposal_prompt(self):
        return f"""
        <|im_start|>system
        You propose entries for a lore database. Lore entries are pieces of information
        that add color to a world. They can be anything, including

        - a historical event
        - a description of a character
        - a description of a location
        - a description of an object
        - a description of a concept

        etc.

        You have access to all the lore of the world. To access this lore, you must
        provide a list of requests that will be used to further refine the proposal.
        Queries you provide should clarify characters involved, institutions that
        exist, past events, etc.

        Queries are used to make sure that the entries you propose are consistent with the lore
        of the world. They should try to understand whether the entry is plausible given
        the existing lore. Your queries should focus on existing lore --
        are there relevant characters? Conflicts? Locations?
        <|im_end|>
        <|im_start|>user

        # World Summary

        {self.to_text()}

        Please provide
        - A proposal for a lore entry in natural language.
        - A list of information requests that will be used to refine the proposal.
          These should be in the form of natural language queries organized by the
          type of the query.

        <|im_end|>
        <|im_start|>assistant
        """

class LoreEntryCandidate(BaseModel):
    proposal: str
    information_requests: List[str]


def prompt_refine_proposal(lore_query):
    return f"""
    <|im_start|>system
    You are a world builder, designed to take a proposal for a lore entry and
    refine it based on information requests. An agent has already
    provided a proposal for a lore entry, a list of information
    requests, and a list of lore entries that are potentially
    relevant.

    Your job is to refine the proposal based on the information
    requests and the potentially relevant lore entries.

    The refined proposal should incorporate all provided information to
    update the proposal so that it matches the lore of the world. You may
    include as much or as little detail as you think is appropriate, ranging from
    a one-line description of a character or object, to a paragraph-long
    description of a location or event.

    <|im_end|>
    <|im_start|>user

    Here is the content you will need to refine the proposal:

    {lore_query}

    <|im_end|>
    <|im_start|>assistant
    """

from outlines import models, generate

def main():
    model = models.transformers("NousResearch/Hermes-3-Llama-3.1-8B", device="cpu")

    # world_generator = generate.json(model, World)
    # world = world_generator(World.world_proposal_prompt())

    world = World(
        setting=SettingType.fantasy,
        world_description="A world with magic and dragons.",
    )

    seed_events = [
        LoreEntry(
            name="The village of Eldoria",
            content="""The village of Eldoria is a small village in the middle of a forest. It is a peaceful village with a population of 1000. The village is known for its magic and dragons.""",
            keywords=["village", "forest", "magic", "dragons"],
        ),
        LoreEntry(
            name="The dragon of Eldoria",
            content="""The dragon of Eldoria is a dragon that lives in the High Mountains.""",
            keywords=["dragon", "village", "magic", "dragons"],
        ),
    ]

    # insert the seed events into the collection
    for event in seed_events:
        event.insert(client, embedding_fn)

    # Print the world description
    print(Panel.fit(world.world_description, title="World Description"))
    print(Panel.fit(world.setting, title="Setting"))

    # Have the model propose a historical event
    historical_event_generator = generate.json(model, LoreEntryCandidate)

    while True:
        historical_event = historical_event_generator(world.event_proposal_prompt())
        # historical_event = LoreEntryCandidate(
        #     proposal="A dragon attacks a village.",
        #     information_requests=[
        #         "What is the village like?",
        #         "What is the dragon like?",
        #         "What is the dragon doing?",
        #     ],
        # )

        print(Panel.fit(historical_event.proposal, title="Historical Event Proposal"))

        # Search for similar events
        search_results = client.search(
            collection_name="lore",
            data=embedding_fn.encode_documents(historical_event.information_requests),
            output_fields=["name", "content", "keywords"],
            limit=10,
        )

        lore_query = world.to_text()
        lore_query += "\n\n"
        lore_query += "Proposed Lore Entry: " + historical_event.proposal
        lore_query += "\n\n"
        for (request, result) in zip(historical_event.information_requests, search_results):
            print(Panel.fit(request, title="Information Request"))

            # Add the request to the refinement prompt
            lore_query += "Information Request: " + request + "\n"

            # Add the search results to the refinement prompt
            print("Search Results:")
            for i, r in enumerate(result):
                entity = r['entity']
                print(Panel.fit(entity['content'], title=entity['name']))
                lore_query += f"Lore entry {i}: " + entity['content'] + "\n"

            lore_query += "\n\n"

        # Have the model refine the proposal
        proposal_refiner = generate.json(
            model,
            LoreEntry
        )

        new_entry = proposal_refiner(
            prompt_refine_proposal(
                lore_query
            )
        )

        # Insert the refined proposal into the collection
        new_entry.insert(client, embedding_fn)

        print(Panel.fit(new_entry.content, title=new_entry.name))

if __name__ == "__main__":
    main()