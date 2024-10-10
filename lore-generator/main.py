from enum import Enum
from typing import List

import openai

from pydantic import BaseModel

import pymilvus

from rich import print
from rich.panel import Panel

from typing import Optional, Dict, Any

def generate(
    client,
    model,
    pydantic_schema,
    prompt,
    system_prompt="You're a helpful assistant."
):
    # Make a request to the local LM Studio server
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format=pydantic_schema
    )

    return response.choices[0].message.parsed


# ███╗   ███╗██╗██╗    ██╗   ██╗██╗   ██╗███████╗
# ████╗ ████║██║██║    ██║   ██║██║   ██║██╔════╝
# ██╔████╔██║██║██║    ██║   ██║██║   ██║███████╗
# ██║╚██╔╝██║██║██║    ╚██╗ ██╔╝██║   ██║╚════██║
# ██║ ╚═╝ ██║██║███████╗╚████╔╝ ╚██████╔╝███████║
# ╚═╝     ╚═╝╚═╝╚══════╝ ╚═══╝   ╚═════╝ ╚══════╝

# ███████╗████████╗██╗   ██╗███████╗███████╗
# ██╔════╝╚══██╔══╝██║   ██║██╔════╝██╔════╝
# ███████╗   ██║   ██║   ██║█████╗  █████╗
# ╚════██║   ██║   ██║   ██║██╔══╝  ██╔══╝
# ███████║   ██║   ╚██████╔╝██║     ██║
# ╚══════╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝

# load the milvus client
client = pymilvus.MilvusClient("milvusdemo-3.db")

# create the collection. default distance metric is "COSINE"
client.create_collection(
    collection_name="lore",
    dimension=768,
    auto_id=True,
)

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
from pymilvus import model as milvus_model
embedding_fn = milvus_model.DefaultEmbeddingFunction()

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
    reasoning_steps: List[str]

    # Embed the content of the lore entry
    def encode(self, embedding_fn):
        return embedding_fn.encode_documents([self.content])[0]

    # Insert the lore entry into the database
    def insert(self, client, embedding_fn):
        # Convert the lore entry to a dictionary and add the embedding
        dict_data = self.model_dump()
        dict_data["vector"] = self.encode(embedding_fn)

        # milvus it, bay-bee
        client.insert(
            collection_name="lore",
            data=[dict_data],
            fields=["content", "reasoning_steps"],
        )

class SettingType(str, Enum):
    """
    The setting of the world. This is an Enum, so
    the language model must choose from one of the
    following options.
    """
    science_fiction = "science fiction"
    fantasy = "fantasy"
    horror = "horror"
    cyberpunk = "cyberpunk"
    steampunk = "steampunk"
    post_apocalyptic = "post apocalyptic"
    magical_realism = "magical realism"


class World(BaseModel):
    setting: SettingType
    world_description: str

    def to_text(self):
        return f"""Genre: {self.setting}\nWorld Description: {self.world_description}"""

    def world_proposal_prompt():
        """
        This function returns the system prompt for the world proposal prompt.

        Returns:
            (system_prompt, user_prompt)
        """

        # System prompt is first, user prompt is second
        return ("""
        You are a world builder. You are given a world description and a setting. Your job is to
        describe a brand new world with a setting and a description of the world.

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
          These should be in the form of natural language queries organized by the
          type of the query.
        """

        return system_prompt, user_prompt

def generate_world(client, model_string):
    system_prompt, user_prompt = World.world_proposal_prompt()
    return generate(
        client=client,
        model=model_string,
        pydantic_schema=World,
        prompt=user_prompt,
        system_prompt=system_prompt
    )

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
    # Go grab whatever the first model we have in LM Studio
    openai_client = openai.OpenAI(
        base_url="http://0.0.0.0:1234/v1",
        api_key="dopeness"
    )
    model = openai_client.models.list().data[0].id

    # Panel display width
    panel_width = 60

    # Generate the world
    world = generate_world(openai_client, model)

    # Print the world description
    print(Panel.fit(world.world_description, title="World Description", width=panel_width))
    print(Panel.fit(world.setting, title="Setting", width=panel_width))

    while True:
        # Propose a historical event
        system_prompt, user_prompt = world.event_proposal_prompt()
        historical_event = generate(
            client=openai_client,
            model=model,
            pydantic_schema=LoreEntryCandidate,
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        # Print out the historical event proposal
        print(Panel.fit(
            historical_event.proposal + \
                "\n\nNumber of information requests: " + \
                str(len(historical_event.information_requests)),
            title="Historical Event Proposal",
            width=panel_width
        ))

        # Ask the model to answer the information requests
        responses = []
        for request in historical_event.information_requests:
            # Print a separator
            print("-" * panel_width)

            # Print the information request, so we know what we're looking for
            print(Panel.fit(
                request,
                title="Information Request",
                width=panel_width
            ))

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
            answer = generate(
                client=openai_client,
                model=model,
                pydantic_schema=InformationRequestAnswer,
                prompt=InformationRequestAnswer.answer_prompt(historical_event.proposal, request, search_results),
                system_prompt=system_prompt
            )

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
        proposal_refiner = generate(
            client=openai_client,
            model=model,
            pydantic_schema=LoreEntry,
            prompt=prompt_refine_proposal(lore_query),
            system_prompt=system_prompt
        )

        # Display reasoning steps
        for step in proposal_refiner.reasoning_steps:
            print(Panel.fit(step, title="Reasoning Step", width=panel_width))

        # Insert the refined proposal into the collection
        proposal_refiner.insert(client, embedding_fn)

        print(Panel.fit(
            proposal_refiner.content,
            title=proposal_refiner.name,
            width=panel_width
        ))

if __name__ == "__main__":
    main()




