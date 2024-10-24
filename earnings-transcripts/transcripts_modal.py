from enum import Enum
import json
from typing import List, Optional

import tqdm
from modal import Image, App, gpu
from rich import print
import os

from pydantic import BaseModel, Field

from transcripts_common import EarningsCall, generate_earnings_calls, load_transcripts, prompt_for_earnings_call, save_earnings_calls

# This creates a modal App object. Here we set the name to "outlines-app".
# There are other optional parameters like modal secrets, schedules, etc.
# See the documentation here: https://modal.com/docs/reference/modal.App
app = App(name="outlines-app")

# Specify a language model to use. This should be the huggingface repo/model name.
LANGUAGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


# Set up the Modal image with the necessary libraries and our huggingface token.
outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.1.1",
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
    "torch",
).env({
    # This will pull in your HF_TOKEN environment variable if you have one.
    'HF_TOKEN':os.environ['HF_TOKEN']
})

# This function imports the model from Hugging Face. The modal container
# will call this function when it starts up. This is useful for
# downloading models, setting up environment variables, etc.
def import_model():
    import outlines
    outlines.models.transformers(
        LANGUAGE_MODEL,
    )

# This line tells the container to run the import_model function when the
# container starts.
outlines_image = outlines_image.run_function(import_model)

# Define a function that uses the image we chose, and specify the GPU
# and memory we want to use.
@app.function(image=outlines_image, gpu=gpu.H100(), timeout=1200)
def generate(
    transcripts: List[str],
):
    import outlines
    model = outlines.models.transformers(
        LANGUAGE_MODEL,
        device="cuda"
    )
    return generate_earnings_calls(model, transcripts)

@app.local_entrypoint()
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
    earnings_calls = generate.remote(transcripts)
    print(f"Generated {len(earnings_calls)} earnings calls")

    # Save the earnings calls
    save_earnings_calls(earnings_calls, transcript_paths)
