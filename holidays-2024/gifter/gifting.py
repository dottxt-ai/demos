"""
.gifter
-----------------------------------
.gifter is a module that generates gift ideas based on the 
description of a recipient. Gift ideas are generated using 
language models guided with structured generation.

Usage Example:
    >>> ideas = generate_gift_ideas("My mom loves gardening and cooking", max_ideas=3)
    >>> print(ideas.gift_ideas[0].name)  # Access results
"""

# Standard library imports
import os
import logging
from enum import Enum
from datetime import datetime
import json

# Third-party imports
import outlines
import torch
from transformers import AutoTokenizer
from pydantic import BaseModel, Field, create_model
from rich import print
from exa_py import Exa
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
# Models listed by increasing resource requirements
MODEL_OPTIONS = {
    "tiny": "HuggingFaceTB/SmolLM2-135M-Instruct",    # Minimal resources
    "small": "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # Balanced choice
    "medium": "NousResearch/Hermes-3-Llama-3.1-8B",    # Better quality
    "large": "meta-llama/Llama-3.3-70B-Instruct"       # Best quality
}

MODEL_STRING = MODEL_OPTIONS["small"]  # Default to balanced option

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('gifting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Structures ---
class GiftType(str, Enum):
    """
    Controlled vocabulary for gift categories.
    Demonstrates use of Enums for type safety and validation.
    """
    BOOK = "book"
    TOY = "toy"
    CLOTHING = "clothing" 
    EXPERIENCE = "experience"  # e.g., tickets, classes, adventures
    ELECTRONICS = "electronics"
    HOME_DECOR = "home_decor"
    FOOD_DRINK = "food_drink"
    SPORTS_OUTDOORS = "sports_outdoors"
    BEAUTY_WELLNESS = "beauty_wellness"
    MUSIC_AUDIO = "music_audio"
    ART_CRAFT = "art_craft"
    JEWELRY = "jewelry"
    GAMES = "games"
    TRAVEL = "travel"
    OTHER = "other"  # Fallback category for unique gifts

class Gift(BaseModel):
    """
    A Gift is a Pydantic class representing a gift idea. 
    
    A Gift has 
    - a type, (e.g. book, toy, clothing, etc, following the enum GiftType)
    - a name
    - a description
    - a reason for the gift
    - a card message written to the recipient
    - a search query to find additional information about the gift

    """
    gift_type: GiftType
    name: str
    description: str
    reason: str
    card_message: str = Field(
        description="A message the gifter should write on the card with the gift",
        default="",
    )
    search_query: str

    def search(self, api_key: str = None):
        """Search for gift-related content using Exa API.
        
        Args:
            api_key: Optional Exa API key (falls back to environment variable)
        """
        logger.debug(f"Searching for: {self.search_query}")
        try:
            if not api_key:
                api_key = os.getenv("EXA_API_KEY")
            
            exa_client = Exa(api_key=api_key)
            result = exa_client.search_and_contents(
                self.search_query,
                type="neural",
                use_autoprompt=True,
                num_results=3,
                highlights=True
            )
            logger.debug(f"Found {len(result.results)} search results")
            return result
        except Exception as e:
            logger.error(f"Error in search for {self.name}: {str(e)}")
            raise

def DynamicGiftIdeas(max_gift_ideas: int = 5, min_gift_ideas: int = 0):
    """
    Factory function demonstrating dynamic Pydantic model creation.
    Shows how to create models with runtime-configured validation.
    """
    return create_model(
        "GiftIdeas",
        person_description=(str, ...),
        gift_reasoning=(str, ...),
        gift_ideas=(
            list[Gift],
            Field(
                max_length=max_gift_ideas, 
                min_length=min_gift_ideas,
                description="List of generated gift suggestions"
            )
        )
    )

# --- Core Generation Logic ---
def setup_model():
    """
    Model initialization demonstrating proper LLM setup and error handling.
    """
    logger.info("Setting up model and tokenizer...")
    try:
        logger.info(f"Using model: {MODEL_STRING}")
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")

        model = outlines.models.transformers(
            MODEL_STRING,
            device="auto",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_STRING)
        logger.info("Model setup complete")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}", exc_info=True)
        raise

MODEL, TOKENIZER = setup_model()
logger.debug("Model and tokenizer loaded successfully")

def template(prompt: str, tokenizer) -> str:
    """
    Prompt templating demonstrating proper LLM input formatting.
    """
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return templated

def generate_gift_ideas(
    person_description: str,
    max_ideas: int = 5,
    min_ideas: int = 0,
) -> 'GiftIdeas':
    """
    Main generation function demonstrating structured LLM output generation.
    Shows how to combine prompting, generation, and validation.
    """
    logger.info(f"Generating {max_ideas} gift ideas for description: {person_description[:50]}...")
    try:
        ConstrainedGiftIdeas = DynamicGiftIdeas(
            max_gift_ideas=max_ideas,
            min_gift_ideas=min_ideas,
        )
        
        gift_ideator = outlines.generate.json(
            MODEL,
            ConstrainedGiftIdeas
        )
        
        # Load prompt template
        with open("prompts/gifts.txt", "r") as f:
            prompt = f.read()
        
        prompt = prompt.format(
            input=person_description,
            schema=ConstrainedGiftIdeas.model_json_schema(),
            max_ideas=max_ideas,
            min_ideas=min_ideas,
        )
        
        ideas = gift_ideator(template(prompt, TOKENIZER))
        return ideas
    except Exception as e:
        logger.error(f"Error generating gift ideas: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    print(".gifter")
    
    prompt = "Describe the gift recipient (e.g., 'My dad loves woodworking and jazz'): \n"
    person_description = input(prompt)
    
    print("\nGenerating gift ideas...")
    ideas = generate_gift_ideas(person_description, max_ideas=3)
    
    print("\nIdeas:")
    print(ideas)
    
    save = input("\nWould you like to save these ideas? (y/n): ").lower()
    if save == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gift_ideas_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(ideas.model_dump(), f, indent=2)
        print(f"\nIdeas saved to {filename}")