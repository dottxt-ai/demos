# sector.py

import hashlib
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

import outlines
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

from rich import print

from exa_py import Exa

# Choose your language model. Phi-3.5-mini-instruct
# should work for most cases, and it's small enough
# that it can run on lots of hardware.
LANGUAGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

# Use transformers for most cases.
MODEL = outlines.models.transformers(
    LANGUAGE_MODEL,
    device="auto",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
)

# Load the tokenizer for adding system/user tokens
TOKENIZER = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)

def to_prompt(user_prompt: str = "", system_prompt: str = "") -> str:
    """Convert user and system prompts to a chat template.

    Args:
        user_prompt: The user's input text
        system_prompt: Optional system instructions

    Returns:
        str: The formatted chat template with special tokens

    Note:
        Outlines does not add special tokens to chat templates,
        so we handle that manually here.
    """
    chat = []

    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})

    if user_prompt:
        chat.append({"role": "user", "content": user_prompt})

    # Convert chat to model's expected format with special tokens
    tokenized = TOKENIZER.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    # Extract the first 10,000 tokens
    tokenized = tokenized[0][:min(len(tokenized[0]), 10000)]

    return TOKENIZER.decode(tokenized)

# Load environment variables from .env file
load_dotenv()

# Load Exa API key from environment variable
api_key = os.getenv("EXA_API_KEY")
if not api_key:
    raise ValueError("EXA_API_KEY environment variable not set")

# Initialize Exa client
exa = Exa(api_key=api_key)

class CustomerType(str, Enum):
    B2B = "B2B"
    B2C = "B2C"

class CompanyInfluence(str, Enum):
    """Enum representing a company's influence level in their market.

    Values:
        LEADER: Market leader/dominant player
        CHALLENGER: Strong competitor challenging the leaders
        FOLLOWER: Company that follows market trends
        NICHE: Specialized player in a market subset
        EMERGING: Growing company with increasing influence
        OTHER: Influence level doesn't fit other categories
    """
    LEADER = "leader"  # Dominant market position
    CHALLENGER = "challenger"  # Strong competitor to leaders
    FOLLOWER = "follower"  # Follows market trends
    NICHE = "niche"  # Specialized in market subset
    EMERGING = "emerging"  # Growing influence
    OTHER = "other"  # Does not fit other categories

class CompanySize(str, Enum):
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"

class CompanyMention(BaseModel):
    name: str
    summary_of_mention: str

class Fact(BaseModel):
    fact: str
    source: str
    reasoning_whether_fact_is_true: List[str]
    whether_to_communicate_fact: bool

class Facts(BaseModel):
    facts: List[Fact]

class MarketPosition(BaseModel):
    company_influence: CompanyInfluence
    company_size: CompanySize
    customer_types: List[CustomerType]
    funding_status: Optional[str]

class NewsSummary(BaseModel):
    quick_summary: str
    key_takeaways: List[str]
    competitors_mentioned: List[str]
    detailed_summary: str

# Analysis of a sector
class CompetitionLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class CompetitiveLandscape(BaseModel):
    competitors: List[str]
    competition_level: CompetitionLevel

class SectorAnalysis(BaseModel):
    competitive_landscape: CompetitiveLandscape
    key_companies: List[str]
    sector_risks: List[str]
    forecast_reasoning: str
    forecast: str

# Search companies
def search_companies(sector: str, search_type: str = "neural", num_results: int = 3, use_autoprompt: bool = True):
    query = exa.search_and_contents(
        f"Find me companies in the {sector} sector",
        type=search_type,
        use_autoprompt=use_autoprompt,
        num_results=num_results,
        category="company",
        text=True
    )

    return [str(result) for result in query.results]

def search_general(query: str, num_results: int = 3):
    query = exa.search_and_contents(
        query,
        type="auto",
        num_results=num_results,
        text=True
    )

    return [str(result) for result in query.results]

# Example multi-step analysis pipeline
def analyze_market_sector(sector: str):
    # 1. Search for companies in sector
    company_queries = search_companies(sector, num_results=3)

    # Quick company extractor function
    company_extractor = outlines.generate.json(MODEL, CompanyMention)
    def extract_companies(company_string: str):
        prompt = to_prompt(
            user_prompt=f"""
            Extract a list of companies in the {sector} sector from the
            following query. Companies should obviously be in the {sector}
            sector. Don't include companies that don't seem to fit in with
            the {sector} sector.

            Provide valid JSON in the schema:
            {CompanyMention.model_json_schema()}

            Query:
            {company_string}
            """
        )

        return company_extractor(prompt)

    companies = [extract_companies(query) for query in company_queries]

    # Ensure we have companies to process
    if not companies:
        print("No companies found in search results")
        return []

    print(companies)

    # Take up to 3 companies
    # max_companies = 3
    # companies = companies[:min(max_companies, len(companies))]

    fact_extractor = outlines.generate.json(MODEL, Facts)
    def extract_facts(company: CompanyMention, results: List[str]):
        return fact_extractor(to_prompt(
            user_prompt=f"""
            Extract a list of facts about the following company. Facts
            should be pieces of useful information about the company,
            pulled from search results. Each fact should be findable in
            the search results. Facts will be used to determine the company's
            market position, key offerings, and other characteristics.

            Some facts may not be true or specifically about the company,
            so please reason whether each fact is true and whether it is
            about the company. Finally, indicate whether each fact should
            be communicated to the user after reviewing your own reasoning,
            the search results, and the schema.

            Company: {company.name}
            Expected schema:
            {Facts.model_json_schema()}

            Search results:
            {results}
            """
        ))

    market_position_extractor = outlines.generate.json(MODEL, MarketPosition)
    def extract_market_position(company: CompanyMention, news_items: List[NewsSummary]):
        return market_position_extractor(to_prompt(
            user_prompt=f"""
            Extract a market position for the following company.
            {MarketPosition.model_json_schema()}

            Company: {company.name}

            News items:
            {news_items}
            """
        ))

    news_summarizer = outlines.generate.json(MODEL, NewsSummary)
    def summarize_news(company: CompanyMention, news_item: str):
        summary = news_summarizer(to_prompt(
            user_prompt=f"""
            Summarize recent news and developments about the following company.
            Focus on significant events, product launches, partnerships,
            financial performance, and market developments that indicate
            the company's trajectory and position.

            Provide a concise but comprehensive summary that gives insight
            into the company's recent history and current direction. I wish
            to use this summary to determine the company's market position,
            competitors, and other characteristics.

            Schema:
            {NewsSummary.model_json_schema()}

            Company: {company.name}

            News item:
            {news_item}
            """
        ))

        print(summary)
        return summary

    # For each company, do additional searches on the company
    all_summaries = []
    for company in companies:
        print(f"Extracting facts for {company.name}")
        search_results = search_general(
            f"Find me information about the {sector} company {company.name}. Want to see \
                information about the company's market position, key offerings, \
                and other characteristics.",
            num_results=3
        )

        summaries = [
            summarize_news(
                company,
                news_item
            ) for news_item in search_results
        ]
        all_summaries.extend(summaries)

        market_position = extract_market_position(company, search_results)
        print(market_position)

    # Go through all summaries and extract market positions
    news_string = "\n".join([str(summary) for summary in all_summaries])

    sector_analyzer = outlines.generate.json(MODEL, SectorAnalysis)
    sector_analysis = sector_analyzer(to_prompt(
        user_prompt=f"""
        Analyze the following news items and companies to determine the
        competitive landscape, key companies, sector risks, and forecast
        for the {sector} sector.

        Schema:
        {SectorAnalysis.model_json_schema()}

        News items:
        {news_string}
        """
    ))
    print(sector_analysis)

    # return analysis

print(analyze_market_sector("AI"))