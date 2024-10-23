from enum import Enum
import json
from typing import List, Optional
from modal import Image, App, gpu
from rich import print
import os

from pydantic import BaseModel, Field

# This creates a modal App object. Here we set the name to "outlines-app".
# There are other optional parameters like modal secrets, schedules, etc.
# See the documentation here: https://modal.com/docs/reference/modal.App
app = App(name="outlines-app")

# Specify a language model to use. This should be the huggingface repo/model name.
language_model = "microsoft/Phi-3.5-mini-instruct"
# language_model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

# Set up the Modal image with the necessary libraries and our huggingface token.
outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.1.1",
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
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
        language_model,
    )

# This line tells the container to run the import_model function when the
# container starts.
outlines_image = outlines_image.run_function(import_model)

class Sentiment(str, Enum):
    """
    Sentiment of the earnings call.
    """
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class AnalystPrediction(str, Enum):
    """
    Analyst prediction of the company's future financial performance.
    """
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"

class FinancialMetrics(BaseModel):
    """
    Financial metrics mentioned in the earnings call. This can be
    extended to include other financial metrics as needed -- just
    add them to the schema.

    We use Optional[thing] for all financial metrics because not all
    earnings calls mention all of these metrics, and forcing the model
    to include them when they do not exist will force the model to
    make numbers up.

    It's useful to also specify units in the schema, otherwise the model
    may use the units specified in the data. These can vary across companies.
    """
    revenue_in_millions: Optional[float] = Field(description="Quarterly revenue in millions of dollars")
    revenue_growth_in_percent: Optional[float] = Field(description="Revenue growth by quarter in percent")
    net_income_in_millions: Optional[float] = Field(description="Quarterly net income in millions of dollars")
    earnings_per_share: Optional[float] = Field(description="Quarterly earnings per share in dollars")
    ebitda_in_millions: Optional[float] = Field(description="Quarterly EBITDA in millions of dollars")
    free_cash_flow_in_millions: Optional[float] = Field(description="Quarterly free cash flow in millions of dollars")

class EarningsCall(BaseModel):
    """
    The main schema for the earnings call analysis. Using outlines to generate
    this schema will extract all the information we request from an earnings
    call transcript.

    To add any new information to the schema, just add a new field to this class
    (or any child classes, like FinancialMetrics).
    """
    company_name: str
    company_ticker: str
    earnings_call_date: str
    earnings_call_quarter: str
    key_takeaways: List[str]

    # Financial metrics
    understanding_of_financial_metrics: str
    financial_metrics: FinancialMetrics

    # Earnings sentiment
    earnings_sentiment: Sentiment

    # Analysis of various risks
    macroeconomic_risk_reasoning: str
    financial_risk_reasoning: str
    operational_risk_reasoning: str
    strategic_risk_reasoning: str

    # Whether the analyst's prediction is a buy, hold, or sell
    investment_recommendation: AnalystPrediction

    # Have the model review its own output for correctness
    review_correctness: List[str]

    # Whether the text must be reprocessed
    text_must_be_reprocessed: bool

# Generate the JSON schema for the EarningsCall class. This is used to tell
# the language model what the output JSON should look like.
schema = json.dumps(EarningsCall.model_json_schema())

# Prompting function
def prompt_for_earnings_call(transcript: str) -> str:
        # <|begin_of_text|>
    return f"""
        <|im_start|>system
        You analyze earnings calls.

        Please begin your review by reading the transcript.

        After this, please

        Identify the company name and ticker symbol.

        Identify the earnings call date and quarter.

        Identify the key takeaways from the call, as a list. This should identify
        key financial information.

        Describe your understanding of the financial metrics mentioned in the call. Which are
        referenced, which are not? Are any metrics referring to other time periods,
        such as year-over-year growth? Are the metrics growth or absolute values?

        Identify the financial metrics mentioned in the call. If no metric is mentioned,
        set the value to null.

        Financial metrics include
        - Quarterly revenue in millions of dollars
        - Revenue growth in percent by quarter
        - Quarterly net income in millions of dollars
        - Earnings per share in dollars
        - EBITDA in millions of dollars
        - Free cash flow in millions of dollars

        Identify the earnings sentiment. This should indicate whether the earnings call
        conveyed generally positive, neutral, or negative information about the company.

        Produce a detailed analysis of the macroeconomic risks that the company faces.
        Review the transcript for any statements by management about macroeconomic risks,
        and use these to produce a detailed analysis of the macroeconomic risks that the
        company faces. Feel free to speculate to the extent that the transcript does not
        provide enough information, or use your own intuition if necessary.

        Describe the firm's exposure to macroeconomic risks, using "low", "medium",
        or "high". The exposure is the extent to which the company is affected by
        macroeconomic risks, and should be informed by the analysis of macroeconomic
        risks. Exposure should take the firm's business model into account, as well
        as any descriptions of current market conditions.

        As if you were a world-class analyst, reason through whether you would recommend
        buying, selling, or holding the company's stock. This should be a list of steps
        that lead to the conclusion, and should include reasoning about the company's
        financial health, macroeconomic risks, and other factors.

        Conclude your recommendation analysis with a single word: "buy", "hold", or "sell".

        Finally, review the JSON output for correctness. Identify any issues with the
        review process, such as incorrect data points, incorrect calculations, etc.

        If any issues are found, set the `text_must_be_reprocessed` field to true.

        You produce output in a valid JSON schema, following this format:

        {schema}

        <|im_end|>
        <|im_start|>user

        Please analyze the following transcript:

        {transcript}

        <|im_end|>
        <|im_start|>assistant
        """

# Define a function that uses the image we chose, and specify the GPU
# and memory we want to use.
@app.function(image=outlines_image, gpu=gpu.H100(), timeout=1200)
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
        device="auto",
    )

    generator = outlines.generate.json(model, schema)

    # For batched inferece
    # earnings_calls = generator([prompt_for_earnings_call(transcript) for transcript in transcripts])

    # One at a time inference
    earnings_calls = [generator(prompt_for_earnings_call(transcript)) for transcript in transcripts]

    # Return the earnings data
    return earnings_calls

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

    # Load all text files in the transcripts directory
    transcripts = []
    transcript_paths = []
    for file_path in transcripts_dir.glob('*.txt'):
        with open(file_path, 'r') as f:
            transcripts.append(f.read())
            transcript_paths.append(str(file_path))

    # DEBUG: only first three
    transcripts = transcripts[:10]
    transcript_paths = transcript_paths[:10]

    earnings_calls = generate.remote(transcripts)

    # Save all earnings calls to a single JSON file
    with open('all_earnings_calls.json', 'w') as f:
        # Convert to json
        json_data = json.dumps(earnings_calls, indent=2)

        # Create a CSV file with selected data from the earnings calls
        import csv

        csv_file_path = 'earnings_summary.csv'

        # Define the CSV headers in lowercase with words separated by underscores
        headers = [
            'ticker', 'company_name', 'quarter', 'analyst_prediction',
            'revenue_m', 'revenue_growth_pct', 'net_income_m', 'eps',
            'ebitda_m', 'free_cash_flow_m', 'earnings_sentiment',
            'needs_reprocessing', 'original_filepath'
        ]

        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            for call, filepath in zip(earnings_calls, transcript_paths):
                writer.writerow({
                    'ticker': call['company_ticker'],
                    'company_name': call['company_name'],
                    'quarter': call['earnings_call_quarter'],
                    'analyst_prediction': call['investment_recommendation'],
                    'revenue_m': call['financial_metrics']['revenue_in_millions'],
                    'revenue_growth_pct': call['financial_metrics']['revenue_growth_in_percent'],
                    'net_income_m': call['financial_metrics']['net_income_in_millions'],
                    'eps': call['financial_metrics']['earnings_per_share'],
                    'ebitda_m': call['financial_metrics']['ebitda_in_millions'],
                    'free_cash_flow_m': call['financial_metrics']['free_cash_flow_in_millions'],
                    'earnings_sentiment': call['earnings_sentiment'],
                    'needs_reprocessing': call['text_must_be_reprocessed'],
                    'original_filepath': filepath
                })

        print(f"CSV file '{csv_file_path}' has been created with the earnings call data.")

        f.write(json_data)

    # Unpack the earnings calls to a csv