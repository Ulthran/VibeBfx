from langchain.schema import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Any, Dict, Sequence
from vibe_bfx.task import Task


class PlanResponse(BaseModel):
    steps: Sequence[str] = Field(
        default_factory=list, description="List of steps to execute"
    )


class RunResponse(BaseModel):
    script: str = Field(description="A bash script to run")
    env: str = Field(
        description="The environment to run the script in (could be venv, conda, docker, etc)"
    )


class Planner:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            timeout=None,
            max_retries=2,
        ).with_structured_output(PlanResponse)

        self.prompt = (
            lambda user_prompt: f"""
            ``` SYSTEM
            You are an expert bioinformatician planning the necessary steps to perform a given task. Try to make each step the equivalent of a single command line tool invocation where possible.

            Also be sure to specify how each step should be mapped to the samples for this project. For example if you need to run `cutadapt` against each sample and then `bwa` against each sample/reference pair.

            Return: A List of Strings, each representing a step to execute.
            ```

            ``` GOOD EXAMPLE RESPONSES
            ["Remove low quality reads and trim with `trimmomatic`. Apply separately to each sample.", "Remove low complexity reads using `github.com/kylebittinger/heyfastq`. Apply separately to each sample.", "Generate a report on the cleaned fastqs using `fastqc`. Apply to all samples to produce an aggregate report."]
            ```

            ``` USER
            {user_prompt}
            ```
        """
        )

    def make_plan(self, prompt: BaseMessage) -> BaseMessage:
        """Generate a sequence of steps to execute based on the prompt."""
        response = self.model.invoke(
            [{"role": "user", "content": self.prompt(prompt.content)}]
        )
        if isinstance(response, dict) and "steps" in response:
            return response["steps"]
        raise ValueError(
            "Invalid response format from model, expected a dictionary with 'steps' key."
        )


class Runner:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            timeout=None,
            max_retries=2,
        ).with_structured_output(RunResponse)

        self.prompt = (
            lambda user_prompt: f"""
            ``` SYSTEM
            You are an expert bioinformatician planning the necessary steps to perform a given task. Try to make each step the equivalent of a single command line tool invocation where possible.

            Also be sure to specify how each step should be mapped to the samples for this project. For example if you need to run `cutadapt` against each sample and then `bwa` against each sample/reference pair.

            Return: A List of Strings, each representing a step to execute.
            ```

            ``` GOOD EXAMPLE RESPONSES
            ["Remove low quality reads and trim with `trimmomatic`. Apply separately to each sample.", "Remove low complexity reads using `github.com/kylebittinger/heyfastq`. Apply separately to each sample.", "Generate a report on the cleaned fastqs using `fastqc`. Apply to all samples to produce an aggregate report."]
            ```

            ``` USER
            {user_prompt}
            ```
        """
        )

    def run(self, prompt: BaseMessage) -> RunResponse:
        response = self.model.invoke(
            [{"role": "user", "content": self.prompt(prompt.content)}]
        )
        return response
