from pymilvus import MilvusClient
from pymilvus import model
from rich import print
from rich.panel import Panel

# Initialize the Milvus client
client = MilvusClient("milvusdemo.db")

# Initialize the embedding function
embedding_fn = model.DefaultEmbeddingFunction()

def semantic_search(query: str, limit: int = 5):
    """
    Perform a semantic search on the 'lore' collection.

    Args:
    query (str): The search query.
    limit (int): The maximum number of results to return.

    Returns:
    list: A list of search results.
    """
    # Embed the query
    query_vector = embedding_fn.encode_documents([query])[0]

    # Perform the search
    results = client.search(
        collection_name="lore",
        data=[query_vector],
        output_fields=["name", "content", "keywords"],
        limit=limit,
    )

    return results[0]  # Return the first (and only) query result

def main():
    # Get collection statistics
    stats = client.get_collection_stats("lore")
    total_records = stats["row_count"]

    print(f"Welcome to the Lore Explorer!")
    print(f"Total records in the database: {total_records}\n")

    while True:
        # Get user input
        query = input("Enter your search query (or 'quit' to exit): ")

        if query.lower() == 'quit':
            break

        # Perform the search
        results = semantic_search(query)

        # Display results
        print(f"\nSearch results for: '{query}'\n")
        for i, result in enumerate(results, 1):
            entity = result['entity']
            print(Panel.fit(
                f"[bold]Name:[/bold] {entity['name']}\n\n"
                f"[bold]Content:[/bold] {entity['content']}\n\n"
                f"[bold]Keywords:[/bold] {', '.join(entity['keywords'])}\n\n"
                f"[bold]Distance:[/bold] {result['distance']}",
                title=f"Result {i}"
            ))
        print("\n")

if __name__ == "__main__":
    main()