from enum import Enum
import json
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from tqdm import tqdm


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


# Prompting function
def prompt_for_earnings_call(transcript: str) -> str:
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

    {EarningsCall.model_json_schema()}

    <|im_end|>
    <|im_start|>user

    Please analyze the following transcript:

    {transcript}

    <|im_end|>
    <|im_start|>assistant
    """

def generate_earnings_calls(language_model, transcripts: List[str], batched: bool = False) -> List[EarningsCall]:
    """
    Generate earnings calls from transcripts.

    Args:
      language_model: The language model to use for generation.
      transcripts: The transcripts to generate earnings calls for.
      batched: Whether to use batched inference. Batched inference is faster
        but more memory intensive. Defaults to False.

    Returns:
      A list of EarningsCall objects.
    """
    import outlines
    model = outlines.models.transformers(
        language_model,
        device="auto",
    )
    generator = outlines.generate.json(model, EarningsCall)

    # For batched inference. This is very VRAM intensive for long earnings calls,
    # so use sparingly.
    if batched:
        earnings_calls = generator([prompt_for_earnings_call(transcript) for transcript in transcripts])
    else:
        # One at a time inference. This is slow but memory efficient.
        earnings_calls = [
            generator(prompt_for_earnings_call(t))
            for t in tqdm(transcripts, desc="Processing transcripts")
        ]

    # Return the earnings data
    return earnings_calls

def load_transcripts(transcripts_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Load all transcripts from a directory. Returns a tuple containing
    the list of transcripts and the list of transcript paths.
    """
    # Load all text files in the transcripts directory
    transcripts = []
    transcript_paths = []
    for file_path in transcripts_dir.glob('*.txt'):
        with open(file_path, 'r') as f:
            transcripts.append(f.read())
            transcript_paths.append(str(file_path))

    return transcripts, transcript_paths

def save_earnings_calls(earnings_calls: List[EarningsCall], transcript_paths: List[str]):
    """
    Save earnings calls to CSV and JSON files.
    """

    # Convert to an array of json objects
    json_data = [earnings_call.model_dump_json(indent=2) for earnings_call in earnings_calls]

    # Save all earnings calls to a single JSON file
    with open('all_earnings_calls.json', 'w') as f:
        f.write(json.dumps(json_data, indent=2))

    # Create a CSV file with selected data from the earnings calls
    import csv

    csv_file_path = 'earnings_summary.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        # Define the CSV headers in lowercase with words separated by underscores
        headers = [
            'ticker', 'company_name', 'quarter', 'analyst_prediction',
            'revenue_m', 'revenue_growth_pct', 'net_income_m', 'eps',
            'ebitda_m', 'free_cash_flow_m', 'earnings_sentiment',
            'needs_reprocessing', 'original_filepath'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for call, filepath in zip(earnings_calls, transcript_paths):
            writer.writerow({
                'ticker': call.company_ticker,
                'company_name': call.company_name,
                'quarter': call.earnings_call_quarter,
                'analyst_prediction': call.analyst_prediction,
                'revenue_m': call.financial_metrics.revenue_in_millions,
                'revenue_growth_pct': call.financial_metrics.revenue_growth_in_percent,
                'net_income_m': call.financial_metrics.net_income_in_millions,
                'eps': call.financial_metrics.earnings_per_share,
                'ebitda_m': call.financial_metrics.ebitda_in_millions,
                'free_cash_flow_m': call.financial_metrics.free_cash_flow_in_millions,
                'earnings_sentiment': call.earnings_sentiment,
                'needs_reprocessing': call.needs_reprocessing,
                'original_filepath': filepath
            })
        print(f"CSV file '{csv_file_path}' has been created with the earnings call data.")