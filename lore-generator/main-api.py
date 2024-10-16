# Misc imports
from enum import Enum
import os
from typing import List
from dotenv import load_dotenv
import openai
from pydantic import BaseModel
import pymilvus
from pymilvus import model as milvus_model
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
import api
import hashlib
import json
import voyageai

load_dotenv(override=True)

# Some stuff for the remote server I'm working with
API_HOST = os.environ.get("DOTTXT_API_HOST", None)
API_KEY = os.environ.get("DOTTXT_API_KEY", None)

# We use Voyage AI for embeddings
vo = voyageai.Client()

# Convenience function to talk to our inference server
def generate(
    pydantic_schema,
    prompt,
):
    generated_text = api.create_completion(pydantic_schema, prompt)
    return generated_text


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
client = pymilvus.MilvusClient("milvusdemo-new.db")

# Remove the collection if it already exists
if client.has_collection("lore"):
    client.drop_collection("lore")

# create the collection. default distance metric is "COSINE"
client.create_collection(
    collection_name="lore",
    dimension=512, # If using voyage-3-lite
    # dimension=768,
    auto_id=True,
)

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
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

    # Embed the content of the lore entry
    def encode(self, embedding_fn):
        # Generate voyage embedding
        result = vo.embed([self.content ], model="voyage-3-lite", input_type="document")
        return result.embeddings[0]

        # return embedding_fn.encode_documents([self.content])[0]

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

    def world_proposal_prompt(world_type=None):
        """
        This function returns the system prompt for the world proposal prompt.
        """

        # System prompt is first, user prompt is second
        return f"""
        <|im_start|>system 
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

        <|im_end|>
        <|im_start|>user

        Please provide a world description and a setting. This should be the seed from which
        the rest of the lore of the world is built.

        Be brief.

        Please provide your world description and setting in a JSON format.
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

        Be brief.

        <|im_end|>
        <|im_start|>user
        # World Summary

        {self.to_text()}

        Please provide:
        - A proposal for a lore entry in natural language.
        - A list of information requests that will be used to refine the proposal.
          These should be in the form of a list of questions to search agents
          that will be used to refine the proposal. Questions you should ask should
          be about various aspects of the world, such as whether a character exists
          to answer the question, or if a location is big enough to fit the action. 
          The goal of your queries should be to help the model refine the proposal.
          Queries should be specific and precise.

        Please provide your proposal and information requests in a JSON format.

        Be very brief and clear.

        <|im_end|>
        <|im_start|>assistant
        """
def generate_world(world_type=None):
    return generate(
        pydantic_schema=World,
        prompt=World.world_proposal_prompt(world_type)
    )

class LoreEntryCandidate(BaseModel):
    proposal: str
    reasoning_steps: List[str]
    information_requests: List[str]

class InformationRequestAnswer(BaseModel):
    reasoning_steps: List[str] # this forces the model to provide reasoning
    answer: str

    def answer_prompt(proposal: str, query: str, search_results: List[LoreEntry]):
        """
        This function returns the system prompt and user prompt for the answer prompt.
        """

        # System prompt is first, user prompt is second
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

        Be brief.

        <|im_end|>
        <|im_start|>user
        Proposal being refined:

        {proposal}

        Information Request from agent:

        {query}

        Search results:

        {search_results}

        Be brief. 

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

    Be brief.

    <|im_end|>
    <|im_start|>user

    Here is the content you will need to refine the proposal:

    {lore_query}

    Provide an in-world description of what occured in the world, using the
    original proposal refined by the queries and answers.

    Be brief. Deliver a short lore entry, no more than 2 sentences.

    <|im_end|>
    <|im_start|>assistant
    """

def separator():
    print("\n")
    print("─" * 60)
    print("\n")

def main():
    # Ask the user what kind of world they want to generate
    print("Tell me what world you want.")

    # Get the world type from the user
    # world_type = "cyberpunk" # if you want a manual world
    world_type = input("> ")

    # Remove the OpenAI client initialization
    # openai_client = openai.OpenAI(
    #     base_url="http://localhost:1234/v1/",
    #     api_key="whatever bro"
    # )
    # model = openai_client.models.list().data[0].id

    # Panel display width
    panel_width = 60

    # Generate the world
    world = generate_world(world_type)

    # Print the world description
    print(Panel.fit(
        world.world_description + "\n\n[italic]Setting: " + world.setting + "[/italic]",
        title="World Description",
        width=panel_width,
        border_style="bright_cyan"
    ))

    # Generate the world forever
    while True:
        # Propose a historical event
        historical_event = generate(
            pydantic_schema=LoreEntryCandidate,
            prompt=world.event_proposal_prompt()
        )

        # Print out the historical event proposal
        separator()
        print(Panel.fit(
            historical_event.proposal + \
                "\n\nNumber of information requests: " + \
                str(len(historical_event.information_requests)),
            title="New proposal",
            width=panel_width,
            border_style="bright_blue"
        ))

        # Ask the model to answer the information requests
        responses = []
        for request in historical_event.information_requests:
            # Print a separator, horizontal line
            separator()

            # Print the information request, so we know what we're looking for
            print(Panel.fit(
                request,
                title="Information Request",
                width=panel_width,
                border_style="red"
            ))

            # Search for similar events in the lore
            search_results = client.search(
                collection_name="lore",
                # data=embedding_fn.encode_documents([request]),
                data=[vo.embed([request], model="voyage-3-lite", input_type="document").embeddings[0]],
                output_fields=["name", "content", "keywords"],
                limit=3,
            )[0]

            # Print out the search results
            for i, result in enumerate(search_results, 1):
                entity = result['entity']
                print(Panel.fit(
                    Markdown(
                        f"**Name:** {entity['name']}\n\n"
                        f"**Content:** {entity['content']}\n\n"
                        f"**Keywords:** {', '.join(entity['keywords'])}\n\n"
                        f"**Distance:** {result['distance']}"
                    ),
                    title=f"Result {i}",
                    width=panel_width,
                    border_style="bright_green"
                ))

            # Have the model answer the information request
            answer_prompt = InformationRequestAnswer.answer_prompt(
                historical_event.proposal,
                request,
                search_results
            )

            answer = generate(
                pydantic_schema=InformationRequestAnswer,
                prompt=answer_prompt
            )

            # Add the answer to the list of answers
            responses.append(answer.answer)

            # Print the reasoning steps and the answer
            for step in answer.reasoning_steps:
                print(Panel.fit(
                    "[italic]" + step + "[/italic]",
                    title="Reasoning Step",
                    width=panel_width,

                ))

            # Print the answer
            print(Panel.fit(answer.answer, title="Answer", width=panel_width))

        lore_query = world.to_text()
        lore_query += "\n\n"
        lore_query += "Proposed Lore Entry: " + historical_event.proposal
        lore_query += "\n\n"
        for (request, response) in zip(historical_event.information_requests, responses):

            # Add the request to the refinement prompt
            lore_query += "Information Request: " + request + "\n"
            lore_query += "Information Answer: " + response + "\n"

            lore_query += "\n\n"

        # Add the system tag
        lore_query = "<|im_start|>system\n" + lore_query + "<|im_end|>\n" + lore_query

        # Have the model refine the proposal
        proposal_refined = generate(
            pydantic_schema=LoreEntry,
            prompt=lore_query
        )

        # how to do this with outlines locally:
        #
        # model = outlines.models.vllm(model_repo, device="cpu")
        #
        # or
        #
        # model = outlines.models.transformers(model_repo, device="cuda")\
        #
        # Generator for lore entry refiner
        # refiner_generator = outlines.generate.json(model, LoreEntry)
        # refined_lore_entry = refiner_generator(system_prompt + lore_query)

        # Insert the refined proposal into the collection
        proposal_refined.insert(client, embedding_fn)

        print(Panel.fit(
            proposal_refined.content,
            title=proposal_refined.name,
            width=panel_width,
            border_style="green"
        ))

if __name__ == "__main__":
    main()
    # print(api.list_schemas())
    # print(to_hash(LoreEntry))
    # print(api.get_completion_endpoint(LoreEntry))
    # print(api.get_completion_url(LoreEntry))
