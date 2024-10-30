# Imports
import outlines
import glob
import pandas as pd
import os
from typing import List
import torch
from transformers import AutoTokenizer
from markdownify import markdownify as md

# Choose your language model
# language_model = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
# language_model = "thesven/Phi-3.5-medium-instruct"
# language_model = "microsoft/Phi-3.5-mini-instruct"
language_model = "meta-llama/Llama-3.1-8B-Instruct"
# language_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
model = outlines.models.transformers(
    language_model,
    device="cuda",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    }
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
        "year": r"(\d{4})",
        "integer": r"(-?\d+)",
        "nullable_integer": r"(-?\d+|null)",
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
    markdown_document = markdown_document.replace("|", "")

    # Split into pages. Pages are separated by horizontal rules in the
    # markdown document
    pages = [page.strip() for page in markdown_document.split("\n---\n")]

    # Print out the first few pages
    for page in pages[0:2]:
        print(page)
        print("\n---\n")

    # Finding the income statement
    yesno = outlines.generate.choice(model, ["Yes", "Maybe", "No"])

    categories = []
    for i in range(len(pages)):
        print(f"\nPage {i}:")
        prompt = to_prompt(
            user_prompt=f"""
            Analyze the following page from a financial filing and determine if it contains
            the consolidated income statement (also known as statement of operations). This should
            be the primary financial statement for the company over the year.

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

            Answer only 'Yes' if this page contains a complete income statement table, or 'No' if it does not.
            If you are not sure, answer 'Maybe'.
            """,
            # system_prompt="You are an expert accountant that locates relevant financial tables in a 10q filing."
        )
        result = yesno(prompt)
        print(result)
        if result == "Yes":
            categories.append(i)

    # Extract the income statement pages and join them with separators
    income_statement_pages = [pages[i] for i in categories]
    income_statement = "\n---\n".join(income_statement_pages)
    print(income_statement)

    # Now we can look at the financial statements and extract the data.
    columns = ["year", "revenue", "operating_income", "net_income"]
    data_types = ["year", "nullable_integer", "nullable_integer", "nullable_integer"]
    csv_pattern = create_regex_pattern(columns, data_types)

    csv_extractor = outlines.generate.regex(
        model,
        csv_pattern,
        sampler=outlines.samplers.multinomial()
    )

    prompt = to_prompt(
        user_prompt=f"""
        Extract annual financial data from this set of pages. Pages
        are from a 10k filing and were chosen because they may contain an income statement.

        Create a row for each year with the following columns: {', '.join(columns)}.

        Extract a row for each year available in the income statement.
        Firms typically report the most recent 3 years of data, but this can vary.

        Each column has types: {', '.join(data_types)}.

        For dollar amounts, use millions, even if the company reports in billions.

        # Relevant pages:

        {income_statement}

        # Key instructions:
        1. Use only annual data, often labeled as "FY" or "12 months ended..."
        2. Use NULL for missing values

        # Output format:

        - CSV format with headers: {','.join(columns)}
        - Use NULL for missing values
        - If no data are found, do not create a row.
        - Enter two newline characters to terminate the CSV when no more data are found.

        # Definitions:
        - Revenue: Total sales of goods and services.
        - Operating income: Revenue minus operating expenses.
        - Net income: Operating income minus taxes.
        """,
        # system_prompt="You extract data from 10k filings and output it in CSV format."
    )

    csv_data = csv_extractor(prompt, max_tokens=500)

    # Create the output dir if it doesn't exist
    os.makedirs("csv", exist_ok=True)

    # Let's save the CSV data to a file
    filename = os.path.basename(file).replace(".html", ".csv")
    with open(f"csv/{filename}", "w") as f:
        f.write(csv_data)

    # Load the extracted data as a dataframe
    df = pd.read_csv(f"csv/{filename}").sort_values(by="year")

    # Load the manual extraction for comparison
    manual_df = pd.read_csv(f"manual/{filename}").sort_values(by="year")

    print(f"Filename: {filename}")
    print("Extracted:")
    print(df.head())
    print("Manual:")
    print(manual_df.head())

    # Compare the two
    valid_matches.append(df.equals(manual_df))
    print(f"Match: {valid_matches[-1]}")

print(f"Total matches: {sum(valid_matches)}/{len(valid_matches)}")
