from enum import Enum
import random
from typing import List

from pydantic import BaseModel
from pymilvus import MilvusClient, FieldSchema, DataType
from pymilvus import model #for embeddings

class Hero(BaseModel):
    name: str
    description: str

class HistoricalEvent(BaseModel):
    heros_involved: List[Hero]
    name: str
    description: str
    year: int

    def encode(self, embedding_fn):
        return embedding_fn.encode_documents([self.name, self.description])

hero_1 = Hero(name="John Doe", description="A hero who is a doctor")
hero_2 = Hero(name="Jane Doe", description="A hero who is a lawyer")

historical_events = [
    HistoricalEvent(
        heros_involved=[hero_1, hero_2],
        name="The Lawsuit of the Century",
        description="A lawsuit between John Doe and Jane Doe",
        year=2024
    ),
    HistoricalEvent(
        heros_involved=[hero_1, hero_2],
        name="The Healthcare Scandal",
        description="Jane heals John, but John sues",
        year=2025
    ),
]

# load the milvus client
client = MilvusClient("milvusdemo.db")

# drop the collection if it already exists,
# since this script bootstraps the databases
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")

# Make the schema for vectors
fields = [
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="nameVector", dtype=DataType.FLOAT_VECTOR, dim=768),
]


# create the collection. default distance metric is "COSINE"
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
# embedding_fn = model.DefaultEmbeddingFunction()

events_dicts = [event.model_dump() for event in historical_events]
# ADD IDS
for i, event in enumerate(events_dicts):
    event["id"] = i
    event["vector"] = [random.random() for _ in range(768)]

client.insert(
    collection_name="demo_collection",
    data=events_dicts,
)

query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]
res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)

# Check for all records
res = client.query(
    collection_name="demo_collection",
    filter="year > 0",
    output_fields=["name", "description", "heros_involved"],
)

print(res)

