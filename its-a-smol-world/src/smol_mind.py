import json
from textwrap import dedent
import outlines
from outlines.samplers import greedy
import re
from transformers import AutoTokenizer, logging
import torch
import io
import contextlib
import warnings


logging.set_verbosity_error()

MODEL_NAME = 'HuggingFaceTB/SmolLM-1.7B-Instruct'
DEVICE = 'mps'
T_TYPE = torch.bfloat16
def format_functions(functions):
    formatted_functions = []
    for func in functions:
        function_info = f"{func['name']}: {func['description']}\n"
        if 'parameters' in func and 'properties' in func['parameters']:
            for arg, details in func['parameters']['properties'].items():
                description = details.get('description', 'No description provided')
                function_info += f"- {arg}: {description}\n"
        formatted_functions.append(function_info)
    return "\n".join(formatted_functions)

SYSTEM_PROMPT_FOR_CHAT_MODEL = dedent("""
    You are an expert designed to call the correct function to solve a problem based on the user's request.
    The functions available (with required parameters) to you are:
    {functions}
    
    You will be given a user prompt and you need to decide which function to call.
    You will then need to format the function call correctly and return it in the correct format.
    The format for the function call is:
    [func1(params_name=params_value]
    NO other text MUST be included.
                                      
    For example:
    Request: I want to order a cheese pizza from Pizza Hut.
    Response: [order_food(restaurant="Pizza Hut", item="cheese pizza", quantity=1)]
                                      
    Request: I want to know the weather in Tokyo.
    Response: [get_weather(city="Tokyo")]

    Request: I need a ride to SFO.
    Response: [order_ride(destination="SFO")]
""")


ASSISTANT_PROMPT_FOR_CHAT_MODEL = dedent("""
    I understand and will only return the function call in the correct format.
    """
)
USER_PROMPT_FOR_CHAT_MODEL = dedent("""
    Request: {user_prompt}. 
""")


def instruct_prompt(question, functions, tokenizer):
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT_FOR_CHAT_MODEL.format(functions=format_functions(functions))},
        {"role": "assistant", "content": ASSISTANT_PROMPT_FOR_CHAT_MODEL },
        {"role": "user", "content": USER_PROMPT_FOR_CHAT_MODEL.format(user_prompt=question)},
    ]
    fc_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return fc_prompt

INTEGER = r"(-)?(0|[1-9][0-9]*)"
STRING_INNER = r'([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])'
# We'll limit this to just a max of 42 characters
STRING = f'"{STRING_INNER}{{1,42}}"'
# i.e. 1 is a not a float but 1.0 is.
FLOAT = rf"({INTEGER})(\.[0-9]+)([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"

simple_type_map = {
    "string": STRING,
    "any": STRING,
    "integer": INTEGER,
    "number": FLOAT,
    "float": FLOAT,
    "boolean": BOOLEAN,
    "null": NULL,
}

def build_dict_regex(props):
    out_re = r"\{"
    args_part = ", ".join(
        [f'"{prop}": ' + type_to_regex(props[prop]) for prop in props]
    )
    return out_re + args_part + r"\}"

def type_to_regex(arg_meta):
    arg_type = arg_meta["type"]
    if arg_type == "object":
        arg_type = "dict"
    if arg_type == "dict":
        try:
            result = build_dict_regex(arg_meta["properties"])
        except KeyError:
            return "Definition does not contain 'properties' value."
    elif arg_type in ["array","tuple"]:
        pattern = type_to_regex(arg_meta["items"])
        result = r"\[(" + pattern + ", ){0,8}" + pattern + r"\]"
    else:
        result = simple_type_map[arg_type]
    return result

type_to_regex({
    "type": "array",
    "items": {"type": "float"}
})

def build_standard_fc_regex(function_data):
    out_re = r"\[" + function_data["name"] + r"\("
    args_part = ", ".join(
        [
            f"{arg}=" + type_to_regex(function_data["parameters"]["properties"][arg])
            for arg in function_data["parameters"]["properties"]

            if arg in function_data["parameters"]["required"]
        ]
    )
    optional_part = "".join(
        [
            f"(, {arg}="
            + type_to_regex(function_data["parameters"]["properties"][arg])
            + r")?"
            for arg in function_data["parameters"]["properties"]
            if not (arg in function_data["parameters"]["required"])
        ]
    )
    return out_re + args_part + optional_part + r"\)]"

def multi_function_fc_regex(fs):
    multi_regex = "|".join([
        rf"({build_standard_fc_regex(f)})" for f in fs
    ])
    return multi_regex

def load_functions(path):
    with open(path, "r") as f:
        return json.load(f)['functions']

class SmolMind:
    def __init__(self, functions, model_name=MODEL_NAME):
        self.functions = functions
        self.fc_regex = multi_function_fc_regex(functions)
        self.model = outlines.models.transformers(
            model_name,
            device=DEVICE,
            model_kwargs={
            "trust_remote_code": True,
            "torch_dtype": T_TYPE,
        })  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = outlines.generate.regex(self.model, self.fc_regex, sampler=greedy())

    def get_function_call(self, user_prompt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prompt = instruct_prompt(user_prompt, self.functions, self.tokenizer)
            response = self.generator(prompt)
        return response
