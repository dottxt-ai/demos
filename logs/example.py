import outlines
import torch
from transformers import AutoTokenizer
from stressed import STRESSED  # Our anxious little helper

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
log_type = "web server"
prompt_template_path = "security-prompt.txt"

# Load the model
model = outlines.models.vllm(
    model_name,
    dtype=torch.bfloat16,
    enable_prefix_caching=True,
    disable_sliding_window=True,
    max_model_len=20000,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize our anxious intern
parser = STRESSED(
    model=model,
    tokenizer=tokenizer,
    log_type=log_type,
    prompt_template_path=prompt_template_path,
    token_max=20000
)

# Load the logs you want to parse
with open("archive/access.log", "r") as file:
    logs = file.readlines()[:1000]

results = parser.analyze_logs(
    logs,
    chunk_size=20,
    format_output=True
)

# Process results (hopefully nothing too alarming)
for analysis in results:
    # Process analysis results
    print(analysis.summary)