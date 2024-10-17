# Standard library imports
import json
import os
from enum import Enum
from typing import List

# Third-party imports
from pydantic import BaseModel, ValidationError
import pymilvus
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from sentence_transformers import SentenceTransformer

# The best package ever
import outlines

# Local imports. Please see loremaster.py for all prompts +
# pydantic classes used by the language model to generate
# results.
from loremaster import (
    InformationRequestAnswer,
    LoreEntry,
    LoreEntryCandidate,
    World,
    prompt_refine_proposal,
    separator,
)

#
# Configurations:
#
# - EMBEDDING_MODEL is the huggingface repo ID associated with your model.
# - EMBEDDING_DEVICE is the device you want to run your embedding model.
#   By default it is "cpu", you may also use "cuda" appropriate.
# - LLM_MODEL is the huggingface repo for the language model you wish to use.
# - LLM_DEVICE is the device to use for the language model. May be "cpu"/"cuda"/etc.
# - PANEL_WIDTH is cosmetic for the bubble displays you'll see in your terminal.
# - SEARCH_RESULT_LIMIT limits how many lore entries are retrieved
#   in each database query. Lower numbers are better for low-context
#   size models, as they may not be able to fit all the lore entries.
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_DEVICE = "cpu"
LLM_MODEL = "microsoft/Phi-3-mini-128k-instruct"
LLM_DEVICE = "cpu"
PANEL_WIDTH = 60
SEARCH_RESULT_LIMIT = 3

# Generate world is a simple convenience function to generate a
# new World object. You may pass a world_type here, which will
# make the language model produce a world related to world_type.
#
# Example: typing "cannibal zombie cyborgs" is likely to produce
# a post-apocalyptic world full of zombie cyborgs.
def generate_world(model, world_type=None):
    # Retrieve the prompt to generate a world.
    system_prompt, user_prompt = World.world_proposal_prompt(world_type)

    # Create a generator function that uses our lanugage model. It will return
    # a World object. Note how simple this is -- no language model nonsense!
    # You just get a World object back.
    generator = outlines.generate.json(model, World)

    # Now we call our generator function with the complete prompt.
    # We're adding system_prompt and user_prompt here. In some inference
    # tools (such as an OpenAI compatible endpoint) you may wish to separate
    # the system and user prompts.
    return generator(system_prompt + user_prompt)

# Let's get this bread
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
    #
    # Now we'll do all the Milvus database stuff.
    # Milvus is a vector database, which means that it
    # stores text in the semantic space a language model
    # understands.
    #
    # When we ask our database queries, we
    # search for content with a similar "vibe" rather
    # than standard keyword search.

    # load the milvus client
    milvus_client = pymilvus.MilvusClient("milvusdemo-new.db")

    # Remove the collection if it already exists.
    # NOTE! this will overwrite any existing lore.
    if milvus_client.has_collection("lore"):
        milvus_client.drop_collection("lore")

    # Set up our embedding model. Nomic is a high-quality embedding
    # model. It has a dimensionality of 768, which we have to specify
    # in
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
        trust_remote_code=True
    )

    # Convenience function to embed text.
    #
    # embed_text("thing")
    #
    # will return an N-vector with the embedding for "thing"
    def embed_text(text):
        return embedding_model.encode([text])[0]

    # Embedding model dimensions can vary when you change the model
    embedding_model_dims = len(embed_text("something"))
    print(f"Embedding model has {embedding_model_dims} elements")

    # create the collection. default distance metric is "COSINE"
    milvus_client.create_collection(
        collection_name="lore",
        dimension=768,
        auto_id=True,
    )

    # Ask the user what kind of world they want to generate
    print("Write something to provide a seed for the world.")

    # Get the world type from the user
    world_seed = input("> ")

    # Set up our outlines model
    model = outlines.models.transformers(
        LLM_MODEL,
        device=LLM_DEVICE
    )

    # Generate the world
    world = generate_world(model, world_seed)

    # Print the world description
    world.print_world_description(PANEL_WIDTH)

    # Historical event generator. This will propose something
    # to add to the lore of the world. It may make no sense,
    # but questions and information from the database will
    # help the language model convert it to something meaningful.
    #
    # historical_event_generator("thing") will return a
    # LoreEntryCandidate object.
    historical_event_generator = outlines.generate.json(
        model,
        LoreEntryCandidate
    )

    # Answer generator is a function that takes a prompt
    # containing search results and returns an answer to
    # a particular query.
    #
    # The result of answer_generator("question") is
    # an InformationRequestAnswer object.
    answer_generator = outlines.generate.json(
        model,
        InformationRequestAnswer
    )

    # proposal_refiner is a function that accepts a prompt and returns
    # a LoreEntry. This function is used to refiner a proposal when
    # given a LoreEntryCandidate and several InformationRequestAnswers.
    #
    # (see use in the while loop for more information)
    proposal_refiner = outlines.generate.json(model, LoreEntry)



    # Generate the world forever.
    # Let the WORLD BUILDING BEGIN
    while True:
        # Propose a historical event
        system_prompt, user_prompt = world.event_proposal_prompt()

        # Pass the world proposal prompt to the historical event
        # generator function. historical_event here is now
        # a LoreEntryCandidate object.
        historical_event = historical_event_generator(
            system_prompt + user_prompt
        )

        # Print out the historical event proposal
        separator()
        historical_event.print_lore_entry_candidate(PANEL_WIDTH)

        # Ask sub-agents to handle the information request answers.
        responses = []
        for request in historical_event.information_requests:
            # Print a separator, horizontal line
            separator()

            # Print the information request, so we know what we're looking for
            print(Panel.fit(
                request,
                title="Information Request",
                width=PANEL_WIDTH,
                border_style="red"
            ))

            # Search for similar events in the lore. This is where
            # we use milvus -- we
            #
            # 1. Embed the search term (request)
            # 2. Retrieve related documents
            search_results = milvus_client.search(
                collection_name="lore",
                data=[embed_text(request)],
                output_fields=["name", "content", "keywords"],
                limit=SEARCH_RESULT_LIMIT
            )[0] # [0] here because we only had one query in data=...

            # Print out the search results
            for i, result in enumerate(search_results, 1):
                # Get the document from the retrieved milvus result
                entity = result['entity']

                # Show a search result on screen.
                print(Panel.fit(
                    Markdown(
                        f"**Name:** {entity['name']}\n\n"
                        f"**Content:** {entity['content']}\n\n"
                        f"**Keywords:** {', '.join(entity['keywords'])}\n\n"
                        f"**Distance:** {result['distance']}"
                    ),
                    title=f"Result {i}",
                    width=PANEL_WIDTH,
                    border_style="bright_green"
                ))

            # Construct a prompt used to ask the language model to
            # answer the lore agent's query.
            system_prompt, user_prompt = InformationRequestAnswer.answer_prompt(
                historical_event.proposal,
                request,
                search_results
            )

            # Make an InformationRequestAnswer
            answer = answer_generator(system_prompt + user_prompt)

            # Add the answer we just got to the list of answers to
            # all the agent's questions.
            responses.append(answer.answer)

            # Print the reasoning steps and the answer
            for step in answer.reasoning_steps:
                print(Panel.fit(
                    "[italic]" + step + "[/italic]",
                    title="Reasoning Step",
                    width=PANEL_WIDTH,

                ))

            # Print the answer the lookup agent provided
            print(Panel.fit(answer.answer, title="Answer", width=PANEL_WIDTH))

        # This part of the code just constructs the final prompt to pass to the
        # refinement prompt. You should think of this as describing the world,
        # the original proposal, the questions the agent asked to determine what
        # lore is relevant to include, and the answers to those questions.
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
        proposal_refined = proposal_refiner(lore_query)

        # Insert the refined proposal into the collection!
        proposal_refined.insert(milvus_client, embed_text)
        proposal_refined.print()

        # Cool. this iteration of the loop has finished,
        #
        # let's
        # do
        # it
        # again
        #

if __name__ == "__main__":
    main()
