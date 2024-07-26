#
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc. Confidential.
#
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
#
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
from re import match

import pandas as pd
import tiktoken
from langchain_openai import AzureChatOpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
from rouge_score import rouge_scorer

# Ideally, we want to return confidence score between 0.0 and 100.0,
# but for ROUGE-1 guard, UI allows the user to configure value between
# 0 and 1, so making scaling factor 1.
SCALING_FACTOR = 1
DEFAULT_OPEN_AI_API_VERSION = "2023-03-15-preview"


def get_token_count(input: str) -> int:
    """Get the token count for the input."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(input, disallowed_special=()))


def get_citation_columns(columns: pd.Index) -> list:
    citation_columns = list(filter(lambda column: match("CITATION_CONTENT_", str(column)), columns))
    return citation_columns


def nemo_response_stage_input_formatter(bot_message: str) -> list:
    """
    Format the input message for the Nemo guard during response guard stage.
    only applicable to bot generated messages.
    this format is only suitable for openai-based nemo guardrails.
    """
    messages = [
        {"role": "context", "content": {"llm_output": bot_message}},
        {"role": "user", "content": "just some place holder message"},
    ]

    return messages


def nemo_response_stage_output_formatter(guard_message: dict) -> str:
    """
    Format the output message for the Nemo guard during response guard stage.
    applicable to nemo guard generated messages.
    this format is only suitable for openai-based nemo guardrails.
    """
    return guard_message["content"]


def get_rouge_1_scorer():
    return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def get_rouge_1_score(
    scorer: rouge_scorer.RougeScorer,
    llm_context: list[str],
    llm_response: list[str],
) -> float:
    """Compute rouge score between list of context sent to LLM and its response.

    Calculate ROUGE score between provided LLM context and LLM's response.
    ROUGE is case insensitive, meaning that upper case letters are treated in same way as lower
    case letters. ROUGE uses a random resampling algorithm which is non-deterministic, so we need
    to fix seed.

    Parameters
    ----------
    llm_context
        context sent from vector database to Open-Source LLM
    llm_response
        confidence score from the Open-Source LLM

    Returns
    -------
        Rouge score between context and the answer
    """
    if (
        llm_response is None
        or len(llm_response) == 0
        or llm_context is None
        or len(llm_context) == 0
    ):
        return 0.0

    valid_llm_responses = list(filter(None, llm_response))
    if len(valid_llm_responses) == 0:
        return 0.0

    # Get only non None contexts for calculation
    valid_llm_contexts = list(filter(None, llm_context))
    if len(valid_llm_contexts) == 0:
        return 0.0

    response_to_score = " ".join(valid_llm_responses)

    # Adapt Greedy Strategy for Maximizing Rouge Score
    # For each sentence keep max between sentence rouge1 precision and sentence rouge1 recall
    # for given llm response. At the end calculate and rouge1 precision and rouge1 recall
    # for the entire block.
    # rouge 1 precision = count of matching n-grams / count of context n-grams
    # rouge 1 recall = count of matching n-grams / count of llm response n-grams
    # According to detailed analysis of ROUGE: https://aclanthology.org/E17-2007.pdf
    # High ROUGE score is hard to achieve, but greedy approacha achieves acceptable results.
    # TODO: https://github.com/Tiiiger/bert_score/ use bert_score instead.
    # Rouge is broken because doesnt' care about semantic only compare token to token
    # We need to capture semantic and this will significantly boost results, because
    # in order to get high rouge, LLM response needs to do "parroting", just mimicking the
    # context as much as possible. Simple GPT paraphrasing with correct answer can break Rouge.

    best_rouge_score = 0.0
    # Greedy Strategy, pick best rouge score between each context sentence and llm response
    for context_sentence in valid_llm_contexts:
        sentence_score = scorer.score(context_sentence, response_to_score)
        best_rouge_score = max(
            best_rouge_score,
            sentence_score["rouge1"].precision,
            sentence_score["rouge1"].recall,
        )

    context_to_score = " ".join(valid_llm_contexts)
    # Compute Rouge between whole context ( concatenated sentences ) and llm response
    block_score = scorer.score(context_to_score, response_to_score)
    best_rouge_score = max(
        best_rouge_score, block_score["rouge1"].precision, block_score["rouge1"].recall
    )
    return best_rouge_score * SCALING_FACTOR


def get_azure_openai_client(
    openai_api_key: str,
    openai_api_base: str,
    openai_deployment_id: str,
) -> AzureChatOpenAI:
    azure_openai_client = AzureChatOpenAI(
        model=openai_deployment_id,
        azure_endpoint=openai_api_base,
        api_key=openai_api_key,
        deployment_name=openai_deployment_id,
        api_version=DEFAULT_OPEN_AI_API_VERSION,
    )
    return azure_openai_client


def calculate_faithfulness(
    evaluator: FaithfulnessEvaluator,
    llm_query: str,
    llm_response: str,
    llm_context: list[str],
):
    """Compute faithfulness score between list of context and LL response for given metric.

    Parameters
    ----------
    llm_query
        query sent from vector database to Open-Source LLM
    llm_response
        response from the Open-Source LLM
    llm_context
        context sent from vector database to Open-Source LLM

    Returns
    -------
        Faithfulness score: 1.0 if the response is faithful to the query, 0.0 otherwise.
    """
    if llm_response is None or llm_query is None or llm_context is None or len(llm_context) == 0:
        return 0.0

    # Get only non None contexts for calculation
    valid_llm_contexts = list(filter(None, llm_context))
    if len(valid_llm_contexts) == 0:
        return 0.0

    faithfulness_result = evaluator.evaluate(llm_query, llm_response, valid_llm_contexts)
    return faithfulness_result.score
