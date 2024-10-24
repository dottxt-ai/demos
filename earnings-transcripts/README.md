# Earnings call analysis

This project uses outlines to extract structured data from earnings call transcripts. Specifying the information you wish to extract is done by specifying a Pydantic model.

## Overview

Many public companies hold earnings calls where they discuss their financial performance and outlook. These calls are transcribed and made available online. Earnings calls are a valuable source of information about a company's financial health and prospects, but are unfortunately not structured data. It can require manual effort to extract the information of interest.

This project uses outlines to extract structured data from earnings call transcripts. All we need to do is specify the information we wish to extract in a Pydantic model, and the language model will extract the data for us.

Data extracted can be hard numbers, such as revenue or net income, or it can be qualitative, such as earnings sentiment. We can even have the model reason about macroeconomic risks the company may face in future quarters.

WARNING! Do not use this as financial advice. This is a proof of concept, and the
output should be verified by a human. Analyzing financial data is extremely difficult
and requires a thorough understanding of the company, the industry, and the
overall economy. Please consult a financial professional before making investment
decisions.

## Usage

This can be run locally (using `transcripts_local.py`) or on [Modal](https://modal.com/) using `transcripts_modal.py`. Modal is a cloud platform supporting GPUs.

### Local

1. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the script:
    ```bash
    python transcripts_local.py
    ```

### Modal

1. [Sign up](https://modal.com/signup) for a Modal account.
2. Install the Modal CLI:
    ```bash
    pip install modal
    ```
2. Set up your Modal key if you have not done so:
    ```bash
    modal setup
    ```
3. Run the script:
    ```bash
    modal run transcripts_modal.py
    ```

1. Download the [data source](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts)
2. Run `unpickle-earnings.py` to extract the transcrips to the `transcripts/` directory. This step adds metadata to each transcript, such as company name, ticker, date, quarter, and the transcript itself.

## How it works

We define a Pydantic model that specifies the information we wish to extract from the transcripts. This model is passed to the language model, which then extracts the data according to the schema.

The schema is defined in `transcripts_common.py`. The default class is `EarningsCall`, which extracts

- Company name and ticker
- Earnings call date and quarter
- Key takeaways from the call, a list of natural-language highlights
- An understanding of the financial metrics mentioned in the call
- Extracted financial metrics in a `FinancialMetrics` object
- Earnings sentiment, whether the call conveyed generally positive, neutral, or negative information about the company
- A detailed analysis of various risks the company faces
    - Macroeconomic risks
    - Financial risks
    - Operational risks
    - Strategic risks
- An investment recommendation, whether to buy, hold, or sell the company
- Review correctness, a self-critique by the language model of the extracted data.
- Whether data needs correction. The model will review the output and note any
  issues it finds. This is useful to identify if the model made up numbers or
  misinterpreted the data.

```python
class Sentiment(str, Enum):
    """
    Sentiment of the earnings call.
    """
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class InvestmentRecommendation(str, Enum):
    """
    Recommendation of whether to buy, hold, or sell the company's stock.
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
    investment_recommendation: InvestmentRecommendation

    # Have the model review its own output for correctness
    review_correctness: List[str]

    # Whether the text must be reprocessed
    text_must_be_reprocessed: bool
```

Financial metrics extracted are

- Revenue
- Revenue growth
- Net income
- Earnings per share
- EBITDA
- Free cash flow

and can easily be extended to include other financial metrics. In the event that you expand the schema to include other financial metrics, you will need to update the prompt to ensure that the model understands the new metrics it needs to extract.

## Limitations

- The model is not perfect. It will sometimes make up numbers or misinterpret the data.
- The model may not understand the data it is extracting if it is not mentioned in the transcripts. For example, if a company announces a new product, but does not discuss its financial impact, the model will not be able to extract that information.

## Future work

- Add a second agent to review the output of the first agent. The `text_must_be_reprocessed` field is currently not used, but could be used to trigger a second agent to attempt a reprocessing of the text.

## Contributing

We welcome contributions to the project! Please open an issue or submit a pull request.

