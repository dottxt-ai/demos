import outlines
import torch
from transformers import AutoTokenizer
from stressed import STRESSED  # Our anxious little helper

# The model we're using
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# The type of logs we're parsing. You don't have to use this, but it's
# helpful for the model to understand the context of the logs.
log_type = "web server"

# The path to the prompt template we're using. This should be a file in
# the repo.
prompt_template_path = "security-prompt.txt"

# Load the model
model = outlines.models.vllm(
    # The model we're using
    model_name,

    # The dtype to use for the model. bfloat16 is faster
    # than the native float size.
    dtype=torch.bfloat16,

    # Enable prefix caching for faster inference
    enable_prefix_caching=True,

    # Disable sliding window -- this is required
    # for prefix caching to work.
    disable_sliding_window=True,

    # The maximum sequence length for the model.
    # Modify this if you have more memory available,
    # and/or if your logs are longer.
    max_model_len=32000,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize our anxious intern!
parser = STRESSED(
    model=model,
    tokenizer=tokenizer,
    log_type=log_type,
    prompt_template_path=prompt_template_path,
    token_max=32000,  # Maximum tokens to generate
    stressed_out=True  # Make the intern more anxious
)

# Load the logs you want to parse. There's three example logs you can
# use in the repo.
test_logs = [
    # Access log for an e-commerce site's web server
    "logs/access-10k.log",
    # Linux system log
    "logs/linux-2k.log",
    # Apache access log
    "logs/apache-2k.log"
]

# Choose the access log for giggles
log_path = test_logs[0]

# Load the logs into memory
with open(log_path, "r") as file:
    logs = file.readlines()

# Start the analysis
results = parser.analyze_logs(
    logs,

    # Chunk the logs into 20 lines at a time.
    # Using a higher number can degenerate the model's performance,
    # but it will generally be faster.
    chunk_size=20,

    # Format output prints a helpful display in your terminal.
    format_output=True
)

# You can do stuff with the results here. results is a list of LogAnalysis objects.
for analysis in results:
    # Do stuff with the analysis
    print(analysis.summary)
