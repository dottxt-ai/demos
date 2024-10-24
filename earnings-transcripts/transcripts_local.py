from enum import Enum
from typing import List, Optional
import outlines
from pathlib import Path

from transcripts_common import (
    EarningsCall,
    generate_earnings_calls,
    prompt_for_earnings_call,
    load_transcripts,
    save_earnings_calls
)

from pydantic import BaseModel, Field



def main():
    # Specify the language model to use
    language_model = "microsoft/Phi-3.5-mini-instruct"

    # Load the language model
    model = outlines.models.transformers(
        language_model,
        device="cuda"
    )

    # Get the directory of the current script
    script_dir = Path(__file__).parent

    # Path to the transcripts directory
    transcripts_dir = script_dir / 'transcripts'

    # Load the transcripts
    transcripts, transcript_paths = load_transcripts(transcripts_dir)

    # Generate the earnings calls
    earnings_calls = generate_earnings_calls(model, transcripts)

    # Save the earnings calls
    save_earnings_calls(earnings_calls, transcript_paths)

if __name__ == "__main__":
    main()
