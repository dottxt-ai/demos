import hashlib
import json
import os
import time
import requests
from dotenv import load_dotenv
from typing import Optional
from requests.exceptions import HTTPError

load_dotenv(override=True)

# Create schema-to-js_id mapping
API_HOST = os.environ.get("DOTTXT_API_HOST", "api.dottxt.co")
API_KEY = os.environ.get("DOTTXT_API_KEY", None)

def check_api_key() -> None:
    if not API_KEY:
        raise ValueError("DOTTXT_API_KEY environment variable is not set")

def get_headers(api_key: Optional[str] = None) -> dict:
    if api_key is None:
        check_api_key()
        api_key = API_KEY
    return {"Authorization": f"Bearer {api_key}"}

SCHEMA_HASH_TO_COMPLETION_URL = {}

def to_hash(pydantic_class):
    schema = pydantic_class.model_json_schema()
    schema_string = json.dumps(schema)
    return hashlib.sha256(schema_string.encode()).hexdigest()

def poll_status(url: str, api_key: Optional[str] = None) -> dict:
    headers = get_headers(api_key)
    while True:
        status_res = requests.get(url, headers=headers)
        status_json = status_res.json()
        if status_res.status_code != 200 or status_json["status"] != "in_progress":
            break
        time.sleep(1)
    return status_json

def get_schema_by_name(name: str, api_key: Optional[str] = None) -> Optional[dict]:
    headers = get_headers(api_key)
    try:
        response = requests.get(f"https://{API_HOST}/v1/json-schemas", headers=headers)
        response.raise_for_status()
        schemas = response.json()['items']
        
        for schema in schemas:
            if schema['name'] == name:
                return schema
        return None
    except HTTPError as e:
        if e.response.status_code == 403:
            raise ValueError("Authentication failed. Please check your API key.") from e
        else:
            raise
    except Exception as e:
        raise


def create_schema(schema: str, name: str, api_key: Optional[str] = None) -> dict:
    data = {"name": name, "json_schema": schema}
    headers = get_headers(api_key)
    try:
        response = requests.post(
            f"https://{API_HOST}/v1/json-schemas", 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        return response.json()
    except HTTPError as e:
        if e.response.status_code == 403:
            raise ValueError("Authentication failed. Please check your API key.") from e
        else:
            raise
    except Exception as e:
        raise


def get_completion_endpoint(model_class, api_key: Optional[str] = None):
    schema_hash = to_hash(model_class)

    if schema_hash in SCHEMA_HASH_TO_COMPLETION_URL:
        completion_url = SCHEMA_HASH_TO_COMPLETION_URL[schema_hash]
        return completion_url
    
    # Check next to see if the schema_has is already stored by checking
    # GET https://api.dottxt.co/v1/json-schemas
    schema_response = get_schema_by_name(schema_hash, api_key)

    # If the schema exists poll the status and return the completion URL
    if schema_response:
        status_url = schema_response["status_url"]
        final_status = poll_status(status_url, api_key)
        completion_url = final_status["completion_url"]
        if completion_url:
            SCHEMA_HASH_TO_COMPLETION_URL[schema_hash] = completion_url
            return completion_url

    # Okay, we don't have a completion URL for this schema. Let's create it.
    schema_string = json.dumps(model_class.model_json_schema())
    schema_response = create_schema(schema_string, schema_hash, api_key)

    # If we get here, we need to wait for the schema to be created
    status_url = schema_response["status_url"]
    final_status = poll_status(status_url, api_key)

    completion_url = final_status["completion_url"]
    if not completion_url:
        raise ValueError(f"No completion URL available for schema: {schema_hash}")

    SCHEMA_HASH_TO_COMPLETION_URL[schema_hash] = completion_url
    return completion_url

def create_completion(model_class, prompt: str, max_tokens: int = 30000, api_key: Optional[str] = None):
    completion_url = get_completion_endpoint(model_class, api_key)
    data = {"prompt": prompt, "max_tokens": max_tokens}
    headers = get_headers(api_key)
    completion_response = requests.post(completion_url, headers=headers, json=data)
    completion_response.raise_for_status()

    # get json
    completion_response_json = completion_response.json()

    # convert to pydantic model
    model = model_class.model_validate_json(completion_response_json['data'])

    return model
