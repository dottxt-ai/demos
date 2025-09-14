import re
import logging
from textwrap import dedent
import outlines
from outlines.samplers import greedy
from transformers import AutoTokenizer, logging as trf_logging
from contextlib import AsyncExitStack
import warnings

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from constants import MODEL_NAME, DEVICE, T_TYPE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smol_mind")
trf_logging.set_verbosity_error()

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
                                      
    Request: Is it raining in NY.
    Response: [get_weather(city="New York")]

    Request: I need a ride to SFO.
    Response: [order_ride(dest="SFO")]
                                      
    Request: I want to send a text to John saying Hello.
    Response: [send_text(to="John", message="Hello!")]
""")


ASSISTANT_PROMPT_FOR_CHAT_MODEL = dedent("""
    I understand and will only return the function call in the correct format.
    """
)
USER_PROMPT_FOR_CHAT_MODEL = dedent("""
    Request: {user_prompt}. 
""")

def continue_prompt(question, functions, tokenizer):
    prompt = SYSTEM_PROMPT_FOR_CHAT_MODEL.format(functions=format_functions(functions))
    prompt += "\n\n"
    prompt += USER_PROMPT_FOR_CHAT_MODEL.format(user_prompt=question)
    return prompt

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

class SmolMind:
    def __init__(self, server_path, model_name=MODEL_NAME, debug=False):
        self.model_name = model_name
        self.debug = debug
        self.server_path = server_path
        self.instruct = True  # Always use instruct mode for MCP
        self.functions = []
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.generator = None
        
        logger.info(f"Initializing model on device: {DEVICE}")
        self.model = outlines.models.transformers(
            model_name,
            device=DEVICE,
            model_kwargs={
                "trust_remote_code": True,
                "torch_dtype": T_TYPE,
            }
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    async def connect_to_server(self):
        """Connect to the MCP server"""
        logger.info(f"Connecting to MCP server: {self.server_path}")
        
        # Determine server type
        is_python = self.server_path.endswith('.py')
        is_js = self.server_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[self.server_path],
            env=None
        )
        
        # Connect to the server using AsyncExitStack
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        # Initialize the session
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        mcp_tools = response.tools
        
        # Convert MCP tools to function format
        self.functions = []
        for tool in mcp_tools:
            func = {
                "name": tool.name,
                "description": tool.description or f"Function {tool.name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Convert input schema to function properties
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                if isinstance(tool.inputSchema, dict):
                    # Extract properties
                    properties = tool.inputSchema.get("properties", {})
                    func["parameters"]["properties"] = properties
                    
                    # Extract required parameters
                    required = tool.inputSchema.get("required", [])
                    func["parameters"]["required"] = required
            
            self.functions.append(func)
        
        # Initialize regex generator
        self.fc_regex = multi_function_fc_regex(self.functions)
        self.generator = outlines.generate.regex(self.model, self.fc_regex, sampler=greedy())
        
        if self.debug:
            tool_names = [tool.name for tool in mcp_tools]
            logger.info(f"Connected to server with tools: {tool_names}")
            
        if not self.functions:
            logger.warning("No functions found from MCP server")
            
        return mcp_tools
    
    async def close(self):
        """Close the connection to the server"""
        await self.exit_stack.aclose()
    
    def get_function_call(self, user_prompt):
        """Generate function call using regex-based generator"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.instruct:
                prompt = instruct_prompt(user_prompt, self.functions, self.tokenizer)
            else:
                prompt = continue_prompt(user_prompt, self.functions, self.tokenizer)
                
            response = self.generator(prompt)
            
            if self.debug:
                logger.info(f"Functions: {self.functions}")
                logger.info(f"Prompt: {prompt}")
                logger.info(f"Generated response: {response}")
                
            return response
    
    async def process_query(self, user_prompt):
        """Process a user query using SmolMind and MCP tools"""
        if not self.functions:
            return "No functions available. Please connect to an MCP server first."
        
        try:
            # Generate the function call using regex generator
            response = self.get_function_call(user_prompt)
            
            # Extract function name and arguments with regex
            match = re.match(r'\[(.*?)\((.*?)\)\]', response)
            if not match:
                return f"Could not parse function call: {response}"
            
            function_name = match.group(1)
            args_str = match.group(2)
            
            # Convert arguments to dictionary
            args_dict = {}
            if args_str:
                # Regex to extract key-value pairs
                pattern = r'(\w+)=("[^"]*"|\'[^\']*\'|\d+|\w+)'
                for key, value in re.findall(pattern, args_str):
                    # Clean string values
                    if value.startswith('"') or value.startswith("'"):
                        value = value[1:-1]
                    # Convert numeric values
                    elif value.isdigit():
                        value = int(value)
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    args_dict[key] = value
            
            # Execute the MCP tool call
            if self.debug:
                logger.info(f"Calling MCP tool: {function_name} with args: {args_dict}")
                
            result = await self.session.call_tool(function_name, args_dict)
            
            return result.content
            
        except Exception as e:
            return f"Error processing query: {str(e)}"