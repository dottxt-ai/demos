# Imports
import re
import outlines
import glob
import pandas as pd
import os
from typing import List
import torch
import tqdm
from transformers import AutoTokenizer
from markdownify import markdownify as md

# Choose your language model
# language_model = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
# language_model = "microsoft/Phi-3-medium-4k-instruct"
language_model = "microsoft/Phi-3.5-mini-instruct"
# language_model = "meta-llama/Llama-3.1-8B-Instruct"
# language_model = "meta-llama/Llama-3.2-3B-Instruct"
# model = outlines.models.transformers(
#     language_model,
#     device="auto",
#     model_kwargs={
#         "torch_dtype": torch.bfloat16,
#     }
# )

model = outlines.models.vllm(
    language_model,
    max_model_len=60000
)

# Load the tokenizer, used for adding system/user tokens
TOKENIZER = AutoTokenizer.from_pretrained(language_model)

def to_prompt(user_prompt="", system_prompt=""):
    """
    Convert a user prompt and system prompt to a chat template.

    Outlines does not add special tokens to the chat template, so we need to
    do this manually.
    """
    chat=[]

    if len(system_prompt) > 0:
        chat.append({'role':'system', 'content':system_prompt})

    if len(user_prompt) > 0:
        chat.append({'role':'user', 'content':user_prompt})

    tokenized_chat = TOKENIZER.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    decoded_chat = TOKENIZER.decode(tokenized_chat[0])
    return decoded_chat

# Example of using the prompt function
# to_prompt(user_prompt="What's up?")

def create_regex_pattern(
    columns: List[str],
    data_types: List[str],
    # The maximum number of rows to extract.
    # Firms usually report the most recent 3 years of data.
    max_rows: int = 3
) -> str:
    # Define regex patterns for common data types
    type_patterns = {
        "string": r"([a-zA-Z\s]+?)",
        "year": r"\d{4}",
        "integer": r"(-?\d+)",
        "integer_comma": r"((-?\d+),?\d+|(\d+))",
        "nullable_integer": r"(-?\d+?|null)",
        "number": r"(-?\d+(?:\.\d{1,2})?)",
        "nullable_number": r"(-?\d+(?:\.\d{1,2})?|null)"
    }

    # Create the header line
    header = ",".join(columns)

    # Create the data capture patterns
    data_patterns = [type_patterns[dtype] for dtype in data_types]
    data_line = ",".join(data_patterns)

    return f"{header}(\n{data_line}){{,{max_rows}}}\n\n"

# Available files
files = glob.glob("10k/*.html")
print(f"Found {len(files)} 10k files")

# Store whether the manual + extraction match
valid_matches = []

# Go through each file
for file in files:
    print(f"Processing {file}")

    # Read the file
    data = open(file, encoding="latin-1").read()

    # Convert to markdown. This removes a lot of the extra HTML
    # formatting that can be token-heavy
    markdown_document = md(data, strip=['a', 'b', 'i', 'u', 'code', 'pre'])

    # Remove table separators
    # markdown_document = markdown_document.replace("|", "")

    # markdown document
    pages = [page.strip() for page in markdown_document.split("\n---\n")]

    # Finding the income statement
    yesno = outlines.generate.choice(model, ["Yes", "Maybe", "No"], sampler=outlines.samplers.greedy())

    categories = []
    for i in tqdm.tqdm(range(len(pages))):
        prompt = to_prompt(
            user_prompt=f"""
            Analyze the following page from a financial filing and determine if it contains
            the comprehensive income statement. This is the primary financial statement for
            the company over the year. Note that this is not the same as the consolidated
            income statement, or the consolidated comprehensive income statement. We want just
            the comprehensive income statement.

            Page Content:
            {pages[i]}

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
            # system_prompt="You are an expert accountant that locates relevant financial tables in a 10q filing."
        )
        result = yesno(prompt)
        if result == "Yes":
            categories.append(i)

    # Extract the income statement pages and join them with separators
    income_statement_pages = [pages[i] for i in categories]
    income_statement = ""
    for i, page in enumerate(income_statement_pages):
        income_statement += f"\n---\nPAGE {i}\n{page}\n---\n"

    # Now we can look at the financial statements and extract the data.
    columns = ["year", "revenue", "operating_income", "net_income"]
    data_types = ["year", "integer_comma", "integer_comma", "integer_comma"]
    csv_pattern = create_regex_pattern(columns, data_types)

    csv_extractor = outlines.generate.regex(
        model,
        csv_pattern,
        sampler=outlines.samplers.greedy()
    )

    raw_text_extractor = outlines.generate.text(model, sampler=outlines.samplers.greedy())

    prompt = to_prompt(
        user_prompt=f"""
        Extract annual financial data from this set of pages. Pages
        are from a 10k filing and were chosen because they may contain
        a comprehensive income statement. Note that selected pages may
        be incorrectly extracted, so you should verify that you are extracting
        from the comprehensive income statement and not some other financial
        statement.

        Create a row for each year available in the income statement with the
        following columns: {', '.join(columns)}. Firms typically report the
        most recent 3 years of data, but this can vary.

        Each column has types: {', '.join(data_types)}.

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

        - CSV format with headers: {','.join(columns)}
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

    # Save the prompt to a file
    os.makedirs("prompts", exist_ok=True)
    with open(f"prompts/{os.path.basename(file).replace('.html', '.txt')}", "w") as f:
        f.write(prompt)

    csv_data = csv_extractor(prompt, max_tokens=500)
    raw_text = raw_text_extractor(prompt, max_tokens=500)

    # Create the output dir if it doesn't exist
    os.makedirs("csv", exist_ok=True)
    os.makedirs("raw", exist_ok=True)

    # Let's save the CSV data to a file
    filename = os.path.basename(file).replace(".html", ".csv")
    with open(f"csv/{filename}", "w") as f:
        f.write(csv_data)

    with open(f"raw/{filename}", "w") as f:
        f.write(raw_text)

    # Load the extracted data as a dataframe
    df = pd.read_csv(f"csv/{filename}").sort_values(by="year")

    # Load the manual extraction for comparison
    manual_df = pd.read_csv(f"manual/{filename}").sort_values(by="year")

    print(f"Filename: {filename}")
    print("Extracted:")
    print(df.head())
    print("Manual:")
    print(manual_df.head())

    # Compare the two dataframes by checking each value directly
    # First ensure both dataframes have the same columns
    df = df[manual_df.columns]

    # Sort both by year to align rows
    df = df.sort_values('year').reset_index(drop=True)
    manual_df = manual_df.sort_values('year').reset_index(drop=True)

    # Compare all values element by element
    matches = (df == manual_df) | (pd.isna(df) & pd.isna(manual_df))
    is_match = matches.all().all()

    valid_matches.append(is_match)
    print(f"Match: {is_match}")

    if not is_match:
        print("Mismatches:")
        for col in df.columns:
            if not matches[col].all():
                print(f"\n{col}:")
                print("Extracted:", df[col].tolist())
                print("Manual:", manual_df[col].tolist())

print(f"Total matches: {sum(valid_matches)}/{len(valid_matches)}")
