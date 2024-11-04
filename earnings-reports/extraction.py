# Imports
from io import StringIO
import outlines
import glob
import pandas as pd
import os
from typing import List
import torch
import tqdm
from transformers import AutoTokenizer
from markdownify import markdownify as md

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

# Regex patterns for the data types
YEAR_REGEX = r"\d{4}"
INTEGER_COMMA_REGEX = r"((-?\d+),?\d+|(\d+))"
NUMBER_REGEX = r"(-?\d+(?:\.\d{1,2})?)"

# Define the column type regex patterns
COLUMN_TYPE_REGEX = {
    "year": YEAR_REGEX,
    "integer_comma": INTEGER_COMMA_REGEX,
    "number": NUMBER_REGEX,
}

# Define the columns to extract, and their data types.
# These are the columns the model
COLUMNS_TO_EXTRACT = {
    "year": "year",
    "revenue": "integer_comma",
    "operating_income": "integer_comma",
    "net_income": "integer_comma",
}

# Load the tokenizer for adding system/user tokens
TOKENIZER = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)

####################################################
#               Convenience functions              #
####################################################


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

    return TOKENIZER.decode(tokenized[0])


def create_regex_pattern(
    # Dictionary mapping column names to their data types
    column_types: dict[str, str],
    # Firms typically report three years of data
    # in the income statement.
    max_rows: int = 3,
) -> str:
    """
    Create a regex pattern for extracting data from a CSV.

    Args:
        column_types: Dictionary mapping column names to their data types
        max_rows: The maximum number of rows to extract.

    Returns:
        str: The regex pattern for the CSV table

    Example:
        {"year": "year", "revenue": "integer_comma"}
        ->
        "year,integer_comma(\n\d{4},((-?\d+),?\d+|(\d+)))"
    """
    # Create the header line. This is the requested column names
    # separated by commas, i.e. "year,revenue,..."
    header = ",".join(column_types.keys())

    # Create the data capture patterns. These are the regex patterns
    # that will be used to capture the data in each column
    data_patterns = [COLUMN_TYPE_REGEX[dtype] for dtype in column_types.values()]
    data_line = ",".join(data_patterns)

    return f"{header}(\n{data_line}){{,{max_rows}}}\n\n"


def load_pages(file: str) -> List[str]:
    """
    Load the pages from a 10k filing.

    Args:
        file: The path to the HTML 10k filing

    Returns:
        List[str]: A list of markdown-formatted pages
    """
    # Read the file in as a string
    data = open(file, encoding="latin-1").read()

    # Convert to markdown. This removes a lot of the extra HTML
    # formatting that can be token-heavy.
    markdown_document = md(data, strip=["a", "b", "i", "u", "code", "pre"])

    # Split the document into pages
    return [page.strip() for page in markdown_document.split("\n---\n")]


def checker_prompt(page: str) -> str:
    return to_prompt(
        user_prompt=f"""
        Analyze the following page from a financial filing and determine if it contains
        the comprehensive income statement. This is the primary financial statement for
        the company over the year.

        Page Content:
        {page}

        Criteria for identification:
        1. Must contain key income statement line items like:
        - Revenue/Sales
        - Cost of Revenue/Cost of Sales
        - Operating Expenses
        - Net Income/Loss
        2. Must show financial results for specific time periods
        3. Must be a primary financial statement (not just discussion or analysis)
        4. Numbers should be presented in a structured tabular format

        Answer only 'Yes' if this page contains a complete comprehensive income statement
        table, or 'No' if it does not. If you are not sure, answer 'Maybe'.
        """,
    )


def find_income_statement(pages: List[str]) -> str:
    # Create a yes/no classifier for identifying income statements
    yesno = outlines.generate.choice(
        MODEL, ["Yes", "Maybe", "No"], sampler=outlines.samplers.greedy()
    )

    # Track which pages contain the income statement
    categories = []
    for i in tqdm.tqdm(range(len(pages))):
        # Result here is one of "Yes", "Maybe", or "No".
        result = yesno(checker_prompt(pages[i]))

        # If the result is "Yes", we've found a page that
        # seems to contain a table related to the income statement.
        if result == "Yes":
            categories.append(i)

    # Extract the income statement-related pages and join them with separators.
    # This will give us one big string we'll hand to the extractor.
    income_statement_pages = [pages[i] for i in categories]
    income_statement = ""
    for i, page in enumerate(income_statement_pages):
        income_statement += f"\n---\nPAGE {i}\n{page}\n---\n"

    return income_statement


def extract_financial_metrics(income_statement: str) -> str:
    # Now we can look at the financial statements and extract the data.
    csv_pattern = create_regex_pattern(COLUMNS_TO_EXTRACT)

    csv_extractor = outlines.generate.regex(
        MODEL, csv_pattern, sampler=outlines.samplers.greedy()
    )

    prompt = to_prompt(
        user_prompt=f"""
        Extract annual financial data from this set of pages. Pages
        are from a 10k filing and were chosen because they may contain
        a comprehensive income statement. Note that selected pages may
        be incorrectly extracted, so you should verify that you are extracting
        from the comprehensive income statement and not some other financial
        statement.

        Create a row for each year available in the income statement with the
        following columns: {', '.join(COLUMNS_TO_EXTRACT.keys())}. Firms typically report the
        most recent 3 years of data, but this can vary.

        Each column has types: {', '.join(COLUMNS_TO_EXTRACT.values())}.

        # Relevant pages:

        {income_statement}

        # Key instructions:

        1. Look ONLY at the "Consolidated Statements of Income" table
        2. For operating income, look for "Income from operations" or "Operating income"
        3. For net income, use the TOTAL net income figure, not amounts allocated to specific share classes
        4. Use NULL for missing values
        5. Operating income must be less than revenue
        6. Net income must be less than operating income
        7. Ignore segment breakdowns, quarterly data, or per-share amounts

        # Output format:

        - CSV format with headers: {','.join(COLUMNS_TO_EXTRACT.keys())}
        - Use NULL for missing values
        - If no data are found, do not create a row.
        - Enter two newline characters to terminate the CSV when no more data are found.

        # Definitions:
        - Revenue: Total sales of goods and services. Usually this is at the top of the
        income statement.
        - Operating income: Revenue minus operating expenses for the entire company. This is revenue
        minus costs. Operating income is also called operating profit, EBIT, or income from
        operations.
        - Net income: Operating income minus taxes. This is the bottom line of the
        income statement.
        """,
        # system_prompt="You extract data from 10k filings and output it in CSV format."
    )

    # Extract our data to a CSV
    csv_data = csv_extractor(prompt, max_tokens=500)

    return csv_data


####################################################
#                  Main function                   #
####################################################
if __name__ == "__main__":
    # Load available 10k filings
    files = glob.glob("10k/*.html")
    print(f"Found {len(files)} 10k files")

    # Go through each file
    for file in files:
        print(f"Processing {file}")

        # Loads the file into a list of markdown pages
        pages = load_pages(file)

        # Find the income statement page. income_statement is a big string
        # of markdown pages that seem to be related to the income statement.
        #
        # Related pages are determined by a checking model that assigns a
        # "Yes" if the page seems to contain the income statement, a "Maybe"
        # if it might contain the income statement, and a "No" otherwise.
        #
        # We take only the "Yes" pages and concatenate them with separators.
        income_statement = find_income_statement(pages)

        # Extract the financial metrics from the income statement.
        #
        # This is a language model call that extracts the financial metrics
        # from the income statement, and returns it in CSV format.
        csv_data = extract_financial_metrics(income_statement)

        # Print out the CSV data in a pandas dataframe
        df = pd.read_csv(StringIO(csv_data))
        print(df)

        # Create the output dir if it doesn't exist
        os.makedirs("csv", exist_ok=True)

        # Save the CSV data to a file
        filename = os.path.basename(file).replace(".html", ".csv")
        with open(f"csv/{filename}", "w") as f:
            f.write(csv_data)
