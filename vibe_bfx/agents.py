import logging
from langchain.schema import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Any, Dict, Sequence
from vibe_bfx.task import Task


class PlanResponse(BaseModel):
    steps: list[str] = Field(
        default_factory=list, description="List of steps to execute"
    )


class RunResponse(BaseModel):
    script: str = Field(description="A bash script to run")
    env: str = Field(
        description="The environment to run the script in (could be venv, conda, docker, etc)"
    )


class ReportResponse(BaseModel):
    summary: str = Field(description="A concise summary of the results")


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

    def make_plan(self, prompt: BaseMessage) -> PlanResponse:
        """Generate a sequence of steps to execute based on the prompt."""
        response: PlanResponse = self.model.invoke(
            [{"role": "user", "content": self.prompt(prompt.content)}]
        )
        print("Planner response: ", response)
        return response


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
        print("Runner response:", response)
        return response


class Reporter:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            timeout=None,
            max_retries=2,
        ).with_structured_output(ReportResponse)

        self.prompt = (
            lambda user_prompt: f"""
            ``` SYSTEM
            You are an expert bioinformatician who has just completed a series of analyses. You have access to the results of these analyses and can summarize them for planning next steps.

            Return: A concise summary of the results, highlighting key findings and their implications.
            ```

            ``` GOOD EXAMPLE RESPONSES
            "The analysis revealed several key insights into the dataset. Notably, we identified a significant increase in gene expression levels in response to the treatment, suggesting a potential mechanism of action. Additionally, the data indicated a correlation between specific genetic markers and observed phenotypic changes, which could inform future research directions."
            ```

            ``` USER
            {user_prompt}
            ```
        """
        )

    def report(self, prompt: BaseMessage) -> BaseMessage:
        response = self.model.invoke(
            [{"role": "user", "content": self.prompt(prompt.content)}]
        )
        print("Reporter response: ", response)
        return response
