"""
😰 STRESSED (STRuctured Generation Security System Evaluating Data) 😰

STRESSED is a system for analyzing security logs using structured generation.
You should think of it as a mildly competent intern reviewing some form
of log for security issues.

    "Everything is fine! This is fine! We're all fine!"

         .-------------------.
       /   Is it supposed to  \
      |    look like that?    |
      |                       |
       \   Maybe I should    /
        \   Google this...  /
         '-------------------'

    Current Status:
    Anxiety         ▓▓▓▓▓▓▓▓▓░ 90%
    Coffee          ▓░░░░░░░░░ 10%
    Understanding   NOT APPLICABLE

    Coping Mechanisms:
    - Deep breaths between log entries
    - Nervous documentation
    - Excessive commenting
    - Strategic panic
"""

# Imports
from enum import Enum
from typing import Literal, Optional
import outlines
from pydantic import BaseModel, Field
from datetime import datetime
import os

# For pretty printing
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text

# Severity levels are used classify the severity of a security event.
# High severity events are those that should be escalated to a human
# for further investigation.
class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

# Attack types are used to classify security events. This is not an exhaustive
# list of attack vectors!
class AttackType(str, Enum):
    BRUTE_FORCE = "BRUTE_FORCE"
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    FILE_INCLUSION = "FILE_INCLUSION"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    UNKNOWN = "UNKNOWN"

# A WebTrafficPattern is a pattern of traffic to a web server --
# it highlights commonly accessed URLs, methods, and response codes.
#
# WebTrafficPatterns are low-priority summarizations used to help
# with understanding the overall traffic patterns to a web server.
class WebTrafficPattern(BaseModel):
    url_path: str
    http_method: str
    hits_count: int
    response_codes: dict[str, int]  # Maps status code to count
    unique_ips: int

# A LogID is a unique identifier for a log entry. The code in this
# script injects a LOGID-<LETTERS> identifier at the beginning of
# each log entry, which we can use to identify the log entry.
# Language models are fuzzy and they often cannot completely
# copy the original log entry verbatim, so we use the LOGID
# to retrieve the original log entry.
class LogID(BaseModel):
    log_id: str = Field(
        description="""
        The ID of the log entry in the format of LOGID-<LETTERS> where
        <LETTERS> indicates the log identifier at the beginning of
        each log entry.
        """,

        # This is a regular expression that matches the LOGID-<LETTERS> format.
        # The model will fill in the <LETTERS> part.
        pattern=r"LOGID-([A-Z]+)",
    )

    # Find the log entry in a list of logs. Simple
    # conveience function.
    def find_in(self, logs: list[str]) -> Optional[str]:
        for log in logs:
            if self.log_id in log:
                return log
        return None

# Class for an IP address.
class IPAddress(BaseModel):
    ip_address: str = Field(
        pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    )

# Class for an HTTP response code.
class ResponseCode(BaseModel):
    response_code: str = Field(
        pattern=r"^\d{3}$",
    )

# A WebSecurityEvent is a security event that occurred on a web server.
#
# WebSecurityEvents are high-priority events that should be escalated
# to a human for further investigation.
class WebSecurityEvent(BaseModel):
    # The log entry IDs that are relevant to this event.
    relevant_log_entry_ids: list[LogID]

    # The reasoning for why this event is relevant.
    reasoning: str

    # The type of event.
    event_type: str

    # The severity of the event.
    severity: SeverityLevel

    # Whether this event requires human review.
    requires_human_review: bool

    # The confidence score for this event. I'm not sure if this
    # is meaningful for language models, but it's here if we want it.
    confidence_score: float = Field(
        ge=0.0, 
        le=1.0,
        description="Confidence score between 0 and 1"
    )

    # Web-specific fields
    url_pattern: str = Field(
        min_length=1,
        description="URL pattern that triggered the event"
    )

    http_method: Literal["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "TRACE", "CONNECT"]
    source_ips: list[IPAddress]
    response_codes: list[ResponseCode]
    user_agents: list[str]

    # Possible attack patterns for this event.
    possible_attack_patterns: list[AttackType]

    # Recommended actions for this event.
    recommended_actions: list[str]

# A LogAnalysis is a high-level analysis of a set of logs.
class LogAnalysis(BaseModel):
    # A summary of the analysis.
    summary: str

    # Observations about the logs.
    observations: list[str]

    # Planning for the analysis.
    planning: list[str]

    # Security events found in the logs.
    events: list[WebSecurityEvent]

    # Traffic patterns found in the logs.
    traffic_patterns: list[WebTrafficPattern]

    # The highest severity event found.
    highest_severity: Optional[SeverityLevel]
    requires_immediate_attention: bool

def format_log_analysis(analysis: LogAnalysis, logs: list[str]):
    """Format a LogAnalysis object into a rich console output.

    Args:
        analysis: A LogAnalysis object (not a list)
        logs: List of original log entries with LOGID prefixes
    """
    console = Console()

    # Create header
    header = Panel(
        f"[bold yellow]Log Analysis Report[/]\n[blue]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        border_style="yellow"
    )

    # Create observations section
    observations = Table(show_header=True, header_style="bold magenta", show_lines=True)
    observations.add_column("Key Observations", style="cyan")
    for obs in analysis.observations:
        observations.add_row(obs)

    # Create security events section
    events_table = Table(show_header=True, header_style="bold red", show_lines=True)
    events_table.add_column("Security Events", style="red")
    events_table.add_column("Details", style="yellow")

    # Create a log table if there are any relevant log entry IDs
    event_logs_table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    event_logs_table.add_column("Related Log Entries", style="cyan", width=100)

    for event in analysis.events:
        event_details = [
            f"Type: {event.event_type}",
            f"Severity: {event.severity.value}",
            f"Confidence: {event.confidence_score * 100}%",
            f"Source IPs: {', '.join([ip.ip_address for ip in event.source_ips])}",
            f"URL Pattern: {event.url_pattern}",
            f"Possible Attacks: {', '.join([attack.value for attack in event.possible_attack_patterns])}"
        ]
        events_table.add_row(
            Text(event.event_type, style="bold red"),
            "\n".join(event_details)
        )

        # Add related logs to the table
        for log_id in event.relevant_log_entry_ids:
            log = log_id.find_in(logs)
            if log:
                event_logs_table.add_row(log)

    # Create traffic patterns section
    traffic_table = Table(show_header=True, header_style="bold green", show_lines=True)
    traffic_table.add_column("URL Path", style="green")
    traffic_table.add_column("Method", style="cyan")
    traffic_table.add_column("Hits", style="yellow")
    traffic_table.add_column("Status Codes", style="magenta")

    for pattern in analysis.traffic_patterns:
        traffic_table.add_row(
            pattern.url_path,
            pattern.http_method,
            str(pattern.hits_count),
            ", ".join(f"{k}: {v}" for k, v in pattern.response_codes.items()),
        )

    # Create summary panel
    summary_text = f"[bold white]Summary:[/]\n[cyan]{analysis.summary}[/]\n\n"
    if analysis.highest_severity:
        summary_text += f"[bold red]Highest Severity: {analysis.highest_severity.value}[/]\n"
    summary_text += f"[bold {'red' if analysis.requires_immediate_attention else 'green'}]" + \
                   f"Requires Immediate Attention: {analysis.requires_immediate_attention}[/]"
    
    summary = Panel(
        summary_text,
        border_style="blue"
    )

    # Print everything
    console.print(header)
    console.print("\n[bold blue]📝 Analysis Summary:[/]")
    console.print(summary)
    console.print(observations)
    console.print("\n[bold red]⚠️  Security Events:[/]")
    console.print(events_table)
    console.print(event_logs_table)
    console.print("\n[bold green]📊 Traffic Patterns:[/]")
    console.print(traffic_table)


class STRESSED:
    def __init__(
        self,
        model,
        tokenizer,
        log_type: str,
        prompt_template_path: str,
        token_max: int,
        stressed_out: bool = False
    ):
        if token_max <= 0:
            raise ValueError("token_max must be positive")
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
        
        self.model = model
        self.tokenizer = tokenizer
        self.log_type = log_type
        self.token_max = token_max
        self.stressed_out = stressed_out
        # Load prompt template
        with open(prompt_template_path, "r") as file:
            self.prompt_template = file.read()

        # Initialize generator
        self.logger = outlines.generate.json(
            self.model,
            LogAnalysis,
            sampler=outlines.samplers.greedy(),
        )

    def _to_prompt(self, text: str, pydantic_class: BaseModel) -> str:
        if self.stressed_out:
            stress_prompt = """
            You are a computer security intern that's really stressed out. 
            Your job is hard and you're not sure you're doing it well.

            Your observations and summaries should reflect your anxiety.
            Convey a sense of urgency and panic, be apologetic, and
            generally act like you're not sure you can do your job.

            In your summary, address your boss as "boss" and apologize for
            any mistakes you've made even if you haven't made any. 

            Use "um" and "ah" a lot.
            """
        else:
            stress_prompt = ""

        messages = []
        
        if self.stressed_out:
            messages.append({"role": "system", "content": stress_prompt})

        messages.append(
            {"role": "user", "content": self.prompt_template.format(
                log_type=self.log_type,
                logs=text,
                model_schema=pydantic_class.model_json_schema(),
                stress_prompt=stress_prompt,
            )}
        )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def analyze_logs(
        self,
        logs: list[str],
        chunk_size: int = 10,
        format_output: bool = True
    ) -> list[LogAnalysis]:
        """
        Analyze a list of log entries.

        Args:
            logs: List of log entries to analyze
            chunk_size: Number of logs to analyze at once
            format_output: Whether to print formatted output

        Returns:
            List of LogAnalysis objects
        """
        results = []

        for i in range(0, len(logs), chunk_size):
            chunked_logs = [log for log in logs[i:i+chunk_size] if log]

            if not chunked_logs:
                continue

            # Create log IDs
            log_ids = [f"LOGID-{chr(65 + (j // 26) % 26)}{chr(65 + j % 26)}"
                      for j in range(len(chunked_logs))]

            logs_with_ids = [f"{log_id} {log}"
                            for log_id, log in zip(log_ids, chunked_logs)]
            chunk = "\n".join(logs_with_ids)

            # Analyze chunk
            prompt = self._to_prompt(chunk, LogAnalysis)
            analysis = self.logger(prompt, max_tokens=self.token_max)

            if format_output:
                format_log_analysis(analysis, logs_with_ids)

            results.append(analysis)

        return results
