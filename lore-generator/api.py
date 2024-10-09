import json
import os
import time
import requests
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

# Create schema-to-js_id mapping
API_HOST = os.environ.get("DOTTXT_API_HOST", "api.dottxt.co")
API_KEY = os.environ.get("DOTTXT_API_KEY", None)

print(API_KEY)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def find_schema(name):
    schemas = list_schemas()
    print(schemas)
    if schemas is None:
        logger.error("Failed to retrieve schemas")
        return None

    if isinstance(schemas, dict) and 'items' in schemas:
        for schema in schemas['items']:
            if isinstance(schema, dict) and schema.get("name") == name:
                return schema
    else:
        logger.error(f"Unexpected schema format: {type(schemas)}")
        return None

    return None

def list_schemas():
    logger.debug(f"API_HOST: {API_HOST}")
    logger.debug(f"API_KEY: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if API_KEY else 'Not set'}")
    logger.debug(f"HEADERS: {HEADERS}")

    try:
        url = f"https://{API_HOST}/v1/json-schemas"
        logger.debug(f"URL: {url}")
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in list_schemas: {e}")
        logger.error(f"Response content: {response.content if 'response' in locals() else 'No response'}")
        return None

def poll_status(url):
    """
    Poll the status of a schema until it's ready.
    """
    print(f"Polling status at {url}")
    while True:
        status_res = requests.get(url, headers=HEADERS)
        if (
            status_res.status_code != 200
            or status_res.json()["status"] != "in_progress"
        ):
            print(f"Status: {status_res.json()}")
            break
        print(f"Status: {status_res.json()}")
        time.sleep(1)

    return status_res.json()

def create_schema(schema, name):
    """
    Create a schema in Dottxt.

    Returns:
    {
        "status_url": "https://api.dottxt.co/json-schemas/js-84885a724a554043aa0edb55d386e759/status",
        "js_id": "js-84885a724a554043aa0edb55d386e759",
        "name": "A flexible schema with string properties",
        "status": "in_progress",
        "detail": null,
        "completion_url": null,
        "created_at": "2024-10-06T21:46:02.603",
        "updated_at": "2024-10-06T21:47:52.603"
    }
    """
    # If we got a Pydantic type, convert it to a JSON schema
    if not isinstance(schema, str):
        schema = json.dumps(schema.model_json_schema())

    data = {"name": name, "json_schema": schema}

    print(data)
    response = requests.post(
        f"https://{API_HOST}/v1/json-schemas", headers=HEADERS, json=data
    )

    # if we got a 500 print the response body
    if response.status_code == 500:
        print(response.text)

    print(response)
    response_json = response.json()

    print(response_json)

    return response_json["js_id"]

def create_completion(url, prompt, max_tokens=20000):
    data = {"prompt": prompt, "seed": 42, "max_tokens": max_tokens}

    # Pretty print the data
    print("Completion data:")
    print(json.dumps(data, indent=2))

    return requests.post(url, headers=HEADERS, json=data)

def get_completion_endpoint(name, model_class):
    # First check to see if the schema exists
    print(f"Getting completion endpoint for {name}")
    schema = find_schema(name)

    if schema is None:
        print(f"Schema not found, creating it")
        # Convert the Pydantic model to a JSON schema
        schema_string = json.dumps(model_class.model_json_schema())

        # Create the schema
        js_id = create_schema(schema_string, name)
        print(f"Schema created: {js_id}")

        # Fetch the full schema details
        schema = find_schema(name)

        if schema is None:
            raise ValueError(f"Failed to retrieve schema after creation: {name}")

    # Now keep checking the status of the schema until it's ready
    status_url = schema["status_url"]
    final_status = poll_status(status_url)

    # Final status now contains the completion URL
    completion_url = final_status["completion_url"]

    if not completion_url:
        raise ValueError(f"No completion URL available for schema: {name}")

    return completion_url
