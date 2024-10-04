from modal import Image, App, gpu
import os

# This creates a modal App object. Here we set the name to "outlines-app".
# There are other optional parameters like modal secrets, schedules, etc.
# See the documentation here: https://modal.com/docs/reference/modal.App
app = App(name="outlines-app")

# Specify a language model to use.
# Another good model to use is "NousResearch/Hermes-2-Pro-Mistral-7B"
language_model = "mistral-community/Mistral-7B-v0.2"

# Please set an environment variable HF_TOKEN with your Hugging Face API token.
# The code below (the .env({...}) part) will copy the token from your local
# environment to the container.
# More info on Image here: https://modal.com/docs/reference/modal.Image
outlines_image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines",
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
).env({
    # This will pull in your HF_TOKEN environment variable if you have one.
    'HF_TOKEN':os.environ['HF_TOKEN']

    # To set the token directly in the code, uncomment the line below and replace
    # 'YOUR_TOKEN' with the HuggingFace access token.
    # 'HF_TOKEN':'YOUR_TOKEN'
})

# This function imports the model from Hugging Face. The modal container
# will call this function when it starts up. This is useful for
# downloading models, setting up environment variables, etc.
def import_model():
    import outlines
    outlines.models.transformers(language_model)

# This line tells the container to run the import_model function when it starts.
outlines_image = outlines_image.run_function(import_model)

# Specify a schema for the character description. In this case,
# we want to generate a character with a name, age, armor, weapon, and strength.
schema = """{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}"""

# Define a function that uses the image we chose, and specify the GPU
# and memory we want to use.
@app.function(image=outlines_image, gpu=gpu.A100(size='80GB'))
def generate(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    # Remember, this function is being executed in the container,
    # so we need to import the necessary libraries here. You should
    # do this with any other libraries you might need.
    import outlines

    # Load the model into memory. The import_model function above
    # should have already downloaded the model, so this call
    # only loads the model into GPU memory.
    model = outlines.models.transformers(
        language_model, device="cuda"
    )

    # Generate a character description based on the prompt.
    # We use the .json generation method -- we provide the
    # - model: the model we loaded above
    # - schema: the JSON schema we defined above
    generator = outlines.generate.json(model, schema)

    # Make sure you wrap your prompt in instruction tags ([INST] and [/INST])
    # to indicate that the prompt is an instruction. Instruction tags can vary
    # by models, so make sure to check the model's documentation.
    character = generator(
        f"<s>[INST]Give me a character description. Describe {prompt}.[/INST]"
    )

    # Print out the generated character.
    print(character)

@app.local_entrypoint()
def main(
    prompt: str = "Amiri, a 53 year old warrior woman with a sword and leather armor.",
):
    # We use the "generate" function defined above -- note too that we are calling
    # .remote() on the function. This tells modal to run the function in our cloud
    # machine. If you want to run the function locally, you can call .local() instead,
    # though this will require additional setup.
    generate.remote(prompt)