"""
The aim of this project is to create a demonstration project that uses
Outlines and local LLMs to create consistent and contextually appropriate
file names for a given directory.

The initial program will assume human readable text files including things like:
- .txt files
- .md files
- .rst files
- .org files
- .json files
- etc.

The program will:
- Scan the given directory and all subdirectories and process all .txt/etc files
- Use the content of all the files in the directory to contextually propose new names
- Rename the files with the new name

"""

import argparse
import os
from pathlib import Path
from textwrap import dedent
import outlines
from outlines.samplers import greedy
import torch
from transformers import AutoTokenizer, logging
from file_proc import create_file_pairs, get_file_metadata, list_supported_files
logging.set_verbosity_error()

MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"

FILE_CATEGORIES = ['data', 'notes', 'meeting', 'memo']

# Example file name:2024-02-29-meeting-planning_next_quarter
FILE_STRUCTURE = r'\d{4}-\d{2}-\d{2}-(' + '|'.join(FILE_CATEGORIES) + r')-[a-zA-Z_]{5,150}'




def create_prompt(file_metadata, tokenizer):
    messages = [
        {
            "role": "user",
            "content": "You are an expert at creating concise and contextually appropriate file names."
        },
        {
            "role": "assistant",
            "content": dedent("""
    I understand. I will rename files according to your instructions, and return only the new name.
    """)
        },
        {
            "role": "user",
            "content": dedent(f"""
    I need you to rename files so it's easy to understand the contents of the file at a glance. The renaming should give
    the date the file was created, the rough category of the file ({", ".join(FILE_CATEGORIES)}) and short summary of the content.
                              
Here is the content of the file I want you to rename (this is only a sample of the initial content of the file):

    ## File content:
    {file_metadata['head_content']}

    ## File creation time (only use this if you can't determine the date from the file content):
    {file_metadata['creation_time']}

    ## End file content
    Please create a new name for this file, and stick to this format:
    - The file name should start with the date in YYYY-MM-DD format.
       - Use information from the file content it self to determine the date, if possible.
       - If there is no date in the file content, use the file creation time.
    - Then specify the type of file using a single word ({", ".join(FILE_CATEGORIES)}). 
    - Finally add a short description but rather than spaces use underscores.
        e.g. "2024-02-29-meeting-planning_next_quarter"
    - Don't worry about the extension, that will be added by another part of the program.
    - ONLY output the new name, nothing else.

    Example:
    2024-02-29-meeting-planning_next_quarter
    """)
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return prompt

def generate_filename(file_content, tokenizer, model):
    generator = outlines.generate.regex(model, FILE_STRUCTURE, sampler=greedy())
    prompt = create_prompt(file_content, tokenizer)
    new_name = generator(prompt)
    
    return new_name

def parse_arguments():
    parser = argparse.ArgumentParser(description="List supported files in a directory.")
    parser.add_argument('--dir', default='.', help="Directory to scan (default: current directory)")
    parser.add_argument('--head-chars', type=int, default=1000, help="Number of characters to read from the head of each file (default: 250)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # List supported files
    files = list_supported_files(args.dir)
    print(f"Preparing to rename these files in {args.dir}:")
    file_metadata = []
    for file in files:
        print(f"{file}:")
        file_metadata.append(get_file_metadata(file, args.head_chars))
    
    print("Generating new names...")
    for i, (file_metadata, _) in enumerate(create_file_pairs(file_metadata)):
        old_path = files[i]
        old_name = os.path.basename(old_path)
        extension = file_metadata['extension']
        new_name_without_extension = generate_filename(file_metadata, tokenizer, model)
        new_name = new_name_without_extension + extension
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        # Rename the file
        try:
            os.rename(old_path, new_path)
            print(f'"{old_name}" -> "{new_name}"')
        except OSError as e:
            print(f"Error renaming file {old_name}: {e}")

if __name__ == "__main__":
    print("loading model")
    model = outlines.models.transformers(
        MODEL_NAME,
        device='mps',
        model_kwargs={
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True
    })
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    main()
