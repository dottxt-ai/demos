# Loremaster 6000

Loremaster 6000 is a tool for generating worlds and historical events in that world.

Loremaster is built to showcase

- [Outlines](https://github.com/dottxt-ai/outlines)
- [Milvus](https://milvus.io/)
- Agent frameworks
- RAG with vector databases

## Features

- Generates a new fictional world with a setting and description.
- Proposes new lore entries for the world, and refines them based on information requests and search results from the Milvus database.
- Inserts the refined lore entries into the Milvus database for future use.
- Provides a rich text-based interface with panels and markdown formatting for displaying the world, lore entries, and search results.

## Requirements

- Python 3.7 or higher
- Packages:
  - `dotenv`
  - `pydantic`
  - `pymilvus`
  - `requests`
  - `rich`
  - `sentence_transformers`
  - `outlines`

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/dottxt-ai/demos.git
   cd demos/lore-generator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the script:
   ```
   python main.py
   ```

   The script will prompt you to provide a seed for the world. After that, it will generate a new world, propose lore entries, and refine them based on the information in the Milvus database.

## How it Works

1. The script sets up the Milvus vector database and the embedding and language models.
2. The user provides a seed for the world, which is used to generate a new fictional world.
3. The script proposes a new lore entry for the world and retrieves relevant information from the Milvus database.
4. The language model refines the lore entry proposal based on the retrieved information and the world description.
5. The refined lore entry is then inserted into the Milvus database.
6. The process repeats, generating new lore entries and refining them based on the existing lore.

The script uses the `outlines` library to generate the world and lore entries, and the `sentence_transformers` library to encode the lore entries for storage in the Milvus database.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

The Loremaster 6000 could easily be made more interactive, more detailed, or more interesting with better prompting/structure/etc.

Go play with it!

## License

This project is licensed under the [MIT License](LICENSE).
