import asyncio
import os
import httpx

from enum import Enum
import random
from typing import List

import httpx
import llama_cpp
import outlines
from pydantic import BaseModel
from pydantic.tools import parse_obj_as
from pymilvus import IndexType, MilvusClient, FieldSchema, DataType
from pymilvus import model #for embeddings

from outlines import models, generate


from rich import print
from rich.panel import Panel

import api

# load the milvus client
client = MilvusClient("milvusdemo-2.db")

# drop the collection if it already exists,
# since this script bootstraps the databases
# if client.has_collection(collection_name="lore"):
    # client.drop_collection(collection_name="lore")

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
    reasoning_steps: List[str]
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
            fields=["content", "reasoning_steps"],
        )

def lore_entry(prompt: str):
    import outlines

    model = outlines.models.transformers(LLM, device="cuda")
    generator = outlines.generate.json(model, LoreEntry)
    return generator(prompt)

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
        describe a brand new world with a setting and a description of the world.

        The setting must be either "fantasy" or "science_fiction". Choose something interesting.

        The world description should be comprehensive. It should describe the unique features of the world,
        focusing primarily on physical properties, geography, and culture.

        Be extremely brief. We'll add more information later.

        <|im_end|>
        <|im_start|>user

        Please provide a world description and a setting. This should be the seed from which
        the rest of the lore of the world is built.
        <|im_end|>
        <|im_start|>assistant
        """

    def event_proposal_prompt(self):
        return f"""
        <|im_start|>system
        You propose entries for a lore database. These should be general plot points,
        characters, locations, etc. Focus on the abstraction like "a hero rises" or
        "the location of a battle".

        Lore entries are pieces of information that add color to a world.
        They can be anything, including

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

def generate_world(model):
    generator = generate.json(model, World)
    return generator(World.world_proposal_prompt())

class LoreEntryCandidate(BaseModel):
    proposal: str
    information_requests: List[str]

class InformationRequestAnswer(BaseModel):
    reasoning_steps: List[str]
    answer: str

    def answer_prompt(proposal: str, query: str, search_results: List[LoreEntry]):
        return f"""
        <|im_start|>system
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

        <|im_end|>
        <|im_start|>user

        Proposal being refined:

        {proposal}

        Information Request from agent:

        {query}

        Search results:

        {search_results}

        Make sure your answer uses only information from the search results.
        Do not make up information.

        <|im_end|>
        <|im_start|>assistant
        """

def prompt_refine_proposal(lore_query: str):
    import outlines

    model = outlines.models.transformers(LLM, device="cuda")
    generator = outlines.generate.json(model, LoreEntry)
    return generator(lore_query)


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

    <|im_end|>
    <|im_start|>assistant
    """


def main():
    # Medium tempterature sampler
    sampler = outlines.samplers.multinomial(temperature=0.5)

    # Panel display width
    panel_width = 60

    # Create a model for the world proposal prompt using llamacpp
    # llm_model = outlines.models.llamacpp(
    #     "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
    #     "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
    #     tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
    #         "NousResearch/Hermes-2-Pro-Llama-3-8B"
    #     ),
    #     n_gpu_layers=-1,
    #     verbose=True,
    # )

    llm_model = outlines.models.transformers(
        "NousResearch/Hermes-2-Pro-Llama-3-8B",
        # tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        #     "NousResearch/Hermes-2-Pro-Llama-3-8B"
        # ),
        device="cpu",
    )

    world = generate_world(llm_model)

    # Print the world description
    print(Panel.fit(world.world_description, title="World Description", width=panel_width))
    print(Panel.fit(world.setting, title="Setting", width=panel_width))

    # Answer generator
    answer_generator = outlines.generate.json(
        llm_model,
        InformationRequestAnswer,
        sampler=sampler
    )

    while True:
        # Have the model propose a historical event
        historical_event_generator = outlines.generate.json(
            llm_model,
            LoreEntryCandidate,
            sampler=sampler
        )

        historical_event = historical_event_generator(world.event_proposal_prompt())
        print(Panel.fit(historical_event.proposal, title="Historical Event Proposal", width=panel_width))

        # Ask the model to answer the information requests
        print("Answering information requests...")
        responses = []
        for request in historical_event.information_requests:
            # Print a separator
            print("-" * panel_width)

            # Print the information request, so we know what we're looking for
            print(Panel.fit(request, title="Information Request", width=panel_width))

            # Search for similar events in the lore
            search_results = client.search(
                collection_name="lore",
                data=embedding_fn.encode_documents([request]),
                output_fields=["name", "content", "keywords"],
                limit=10,
            )[0]

            # Print out the search results
            print("Search results:")
            for i, result in enumerate(search_results, 1):
                entity = result['entity']
                print(Panel.fit(
                    f"[bold]Name:[/bold] {entity['name']}\n\n"
                    f"[bold]Content:[/bold] {entity['content']}\n\n"
                    f"[bold]Keywords:[/bold] {', '.join(entity['keywords'])}\n\n"
                    f"[bold]Distance:[/bold] {result['distance']}",
                    title=f"Result {i}",
                    width=panel_width,
                ))

            # Have the model answer the information request
            answer = answer_generator(InformationRequestAnswer.answer_prompt(historical_event.proposal, request, search_results))

            # Add the answer to the list of answers
            responses.append(answer.answer)

            # Print the reasoning steps and the answer
            print("Reasoning steps:")
            for step in answer.reasoning_steps:
                print(Panel.fit(step, title="Reasoning Step", width=panel_width))

            # Print the answer
            print(Panel.fit(answer.answer, title="Information Request Answer", width=panel_width))

        print("Constructing lore query...")

        lore_query = world.to_text()
        lore_query += "\n\n"
        lore_query += "Proposed Lore Entry: " + historical_event.proposal
        lore_query += "\n\n"
        for (request, response) in zip(historical_event.information_requests, responses):

            # Add the request to the refinement prompt
            lore_query += "Information Request: " + request + "\n"
            lore_query += "Information Answer: " + response + "\n"

            lore_query += "\n\n"

        # Have the model refine the proposal
        proposal_refiner = generate.json(
            llm_model,
            LoreEntry
        )

        print("Refining proposal...")
        new_entry = proposal_refiner(
            prompt_refine_proposal(
                lore_query
            )
        )

        # Display reasoning steps
        print("Reasoning steps...")
        for step in new_entry.reasoning_steps:
            print(Panel.fit(step, title="Reasoning Step", width=panel_width))

        # Insert the refined proposal into the collection
        print("Inserting refined proposal...")
        new_entry.insert(client, embedding_fn)

        print(Panel.fit(new_entry.content, title=new_entry.name, width=panel_width))

if __name__ == "__main__":
    main()

    # Make schema dir if it doesn't exist
    # if not os.path.exists("schemas"):
    #     os.makedirs("schemas")

    # asyncio.run(main_async())




