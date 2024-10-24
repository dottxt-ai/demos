from enum import Enum
import json
import csv
from typing import List, Optional
from rich import print
import outlines
import os

from transcripts_common import *

from pydantic import BaseModel, Field

language_model = "meta-llama/Llama-3.2-1B"

model = outlines.models.transformers(
    language_model,
)


# Define a function that uses the image we chose, and specify the GPU
# and memory we want to use.
def generate(
    transcripts: List[str],
):
    # Remember, this function is being executed in the container,
    # so we need to import the necessary libraries here. You should
    # do this with any other libraries you might need.
    import outlines

    # Load the model into memory. The import_model function above
    # should have already downloaded the model, so this call
    # only loads the model into GPU memory.
    model = outlines.models.transformers(
        language_model,
        device="cuda",
    )

    generator = outlines.generate.json(model, EarningsCall)

    # For batched inferece
    # earnings_calls = generator([prompt_for_earnings_call(transcript) for transcript in transcripts])

    # One at a time inference with progress bar
    from tqdm import tqdm
    earnings_calls = [
        generator(prompt_for_earnings_call(transcript))
        for transcript in tqdm(transcripts, desc="Processing transcripts")
    ]

    # Return the earnings data
    return earnings_calls

def main(
):
    import os
    import json
    from pathlib import Path

    # Get the directory of the current script
    script_dir = Path(__file__).parent

    # Path to the transcripts directory
    transcripts_dir = script_dir / 'transcripts'

    # Load the transcripts
    transcripts, transcript_paths = load_transcripts(transcripts_dir)

    # Generate the earnings calls
    earnings_calls = generate_earnings_calls(language_model, transcripts)

    # Save the earnings calls
    save_earnings_calls(earnings_calls, transcript_paths)

if __name__ == "__main__":
    main()
