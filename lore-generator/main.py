# Standard library imports
import json
import os
from enum import Enum
from typing import List

# Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import pymilvus
import requests
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from sentence_transformers import SentenceTransformer
import outlines

# Local imports
from loremaster import (
    InformationRequestAnswer,
    LoreEntry,
    LoreEntryCandidate,
    World,
    prompt_refine_proposal,
    separator,
)

def generate_world(model, world_type=None):
    system_prompt, user_prompt = World.world_proposal_prompt(world_type)
    generator = outlines.generate.json(model, World)
    return generator(system_prompt + user_prompt)

def main():
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
    milvus_client = pymilvus.MilvusClient("milvusdemo-new.db")

    # Remove the collection if it already exists
    if milvus_client.has_collection("lore"):
        milvus_client.drop_collection("lore")

    # create the collection. default distance metric is "COSINE"
    milvus_client.create_collection(
        collection_name="lore",
        dimension=768,
        auto_id=True,
    )

    # Set up our embedding model
    embedding_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        device="cpu",
        trust_remote_code=True
    )

    # Convenience function to embed text
    def embed_text(text):
        return embedding_model.encode([text])[0]

    # Ask the user what kind of world they want to generate
    print("Describe what kind of world you want to generate.")

    # Get the world type from the user
    # world_seed = input("> ")
    world_seed = "science_fiction"

    # Set up our outlines model
    model = outlines.models.transformers(
        "microsoft/Phi-3-mini-128k-instruct",
        device="cpu"
    )

    # Panel display width
    panel_width = 60

    # Generate the world
    world = generate_world(model, world_seed)

    # Print the world description
    print(Panel.fit(
        world.world_description + "\n\n[italic]Setting: " + world.setting + "[/italic]",
        title="World Description",
        width=panel_width,
        border_style="bright_cyan"
    ))

    # Historical event generator
    historical_event_generator = outlines.generate.json(model, LoreEntryCandidate)

    # Generate the world forever
    while True:
        # Propose a historical event
        system_prompt, user_prompt = world.event_proposal_prompt()
        historical_event = historical_event_generator(system_prompt + user_prompt)
        # historical_event = generate(
        #     client=openai_client,
        #     model=model,
        #     pydantic_schema=LoreEntryCandidate,
        #     prompt=user_prompt,
        #     system_prompt=system_prompt
        # )

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
                # style="on grey0"
                border_style="red"
            ))

            # Search for similar events in the lore
            search_results = milvus_client.search(
                collection_name="lore",
                data=[embed_text(request)],
                output_fields=["name", "content", "keywords"],
                limit=10,
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
            system_prompt, user_prompt = InformationRequestAnswer.answer_prompt(
                historical_event.proposal,
                request,
                search_results
            )

            answer_generator = outlines.generate.json(model, InformationRequestAnswer)
            answer = answer_generator(system_prompt + user_prompt)
            # answer = generate(
            #     client=openai_client,
            #     model=model,
            #     pydantic_schema=InformationRequestAnswer,
            #     prompt=user_prompt,
            #     system_prompt=system_prompt
            # )

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

        # Have the model refine the proposal
        proposal_refiner = outlines.generate.json(model, LoreEntry)
        proposal_refined = proposal_refiner(lore_query)
        # proposal_refined = generate(
        #     client=openai_client,
        #     model=model,
        #     pydantic_schema=LoreEntry,
        #     prompt=prompt_refine_proposal(lore_query),
        #     system_prompt=system_prompt
        # )

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
        proposal_refined.insert(milvus_client, embed_text)

        print(Panel.fit(
            proposal_refined.content,
            title=proposal_refined.name,
            width=panel_width,
            border_style="green"
        ))

if __name__ == "__main__":
    main()
