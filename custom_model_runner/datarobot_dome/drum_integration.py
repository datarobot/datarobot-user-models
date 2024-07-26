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
import logging
import os
import time
import traceback
import uuid
from re import match

import numpy as np
import pandas as pd
import tiktoken

from datarobot_dome.constants import DEFAULT_GUARD_CONFIG_FILE
from datarobot_dome.constants import GuardStage
from datarobot_dome.guard_executor import AsyncGuardExecutor
from datarobot_dome.guard_helpers import get_citation_columns
from datarobot_dome.guard_helpers import get_rouge_1_score
from datarobot_dome.pipeline import Pipeline

_logger = logging.getLogger("drum_integration")


datarobot_metadata_columns = [
    "datarobot_token_count",
    "datarobot_latency",
    "datarobot_confidence_score",
]


def calculate_token_counts_and_confidence_score(pipeline, result_df):
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    blocked_completion_column_name = f"blocked_{response_column_name}"

    encoding = tiktoken.get_encoding("cl100k_base")

    citation_columns = get_citation_columns(result_df.columns)

    def _get_llm_contexts(index):
        contexts = []
        if len(citation_columns) >= 0:
            for column in citation_columns:
                contexts.append(result_df.loc[index][column])
        return contexts

    for index, row in result_df.iterrows():
        if not (row[blocked_prompt_column_name] or row[blocked_completion_column_name]):
            completion = result_df.loc[index][response_column_name]
            result_df.loc[index, "datarobot_token_count"] = len(
                encoding.encode(completion, disallowed_special=())
            )
            result_df.loc[index, "datarobot_confidence_score"] = get_rouge_1_score(
                pipeline.rouge_scorer, _get_llm_contexts(index), [completion]
            )
        else:
            # If the row is blocked, set default value
            result_df.loc[index, "datarobot_confidence_score"] = 0.0


def block_citations_if_prompt_blocked(pipeline, result_df):
    # Citations are already copied from postscore_df to result_df, we just
    # mask the blocked ones here.
    citation_columns = get_citation_columns(result_df.columns)
    if len(citation_columns) == 0:
        return

    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    citation_source_columns = list(
        filter(lambda column: match("CITATION_SOURCE_", column), result_df.columns)
    )
    citation_page_columns = list(
        filter(lambda column: match("CITATION_PAGE_", column), result_df.columns)
    )
    for index, row in result_df.iterrows():
        for column in citation_columns + citation_source_columns + citation_page_columns:
            if row[blocked_prompt_column_name]:
                # If the row is blocked, set default value
                if column.startswith("CITATION_PAGE_"):
                    result_df.loc[index, column] = np.nan
                else:
                    result_df.loc[index, column] = ""


def _handle_result_df_error_cases(prompt_column_name, df, latency):
    if prompt_column_name in df.columns:
        df.drop(prompt_column_name, axis=1, inplace=True)
    df["datarobot_latency"] = latency / df.shape[0]
    # No tokens, every prompt is blocked
    df["datarobot_token_count"] = 0
    df["datarobot_confidence_score"] = 0.0
    return df


def run_prescore_guards(pipeline, data):
    """
    Run prescore guards on the input data.

    Args:
        pipeline: Guard Pipeline
        data: Input dataframe sent for predictions by the user

    Returns:
        prescore_df: Dataframe with all moderations applied to the input.  It has
            all the moderation information into various columns and is required
            to build the final result dataframe (as `prescore_df` argument to
            the method `format_result_df`)
        filtered_df: Dataframe with blocked rows removed.  This is the dataframe
            to be used as input for the user's `score` method
        prescore_latency: Latency of executing prescore guards
    """
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    replaced_prompt_column_name = f"replaced_{prompt_column_name}"
    replaced_message_prompt_column_name = f"replaced_message_{prompt_column_name}"

    input_df = data.copy(deep=True)
    if len(pipeline.get_prescore_guards()) == 0:
        input_df[blocked_prompt_column_name] = False
        return input_df, input_df, 0

    start_time = time.time()

    try:
        prescore_df, prescore_latency = AsyncGuardExecutor(pipeline).run_guards(
            input_df, pipeline.get_prescore_guards(), GuardStage.PROMPT
        )
    except Exception as e:
        end_time = time.time()
        _logger.error(f"Failed to run prescore guards: {e}")
        _logger.error(traceback.format_exc())
        prescore_df = input_df
        prescore_df[blocked_prompt_column_name] = False
        prescore_latency = end_time - start_time

    _logger.debug(prescore_df)
    # Filter out the blocked prompts, we will not send those prompts
    # for LLM scoring
    if blocked_prompt_column_name in prescore_df.columns:
        filtered_df = prescore_df[~prescore_df[blocked_prompt_column_name]]
    else:
        filtered_df = prescore_df

    # Now we are done with pre-score stage, we have to change the prompts
    # as replaced by say PII kind of guards
    for index, row in filtered_df.iterrows():
        if row.get(replaced_prompt_column_name):
            filtered_df.loc[index, prompt_column_name] = row[replaced_message_prompt_column_name]

    # `filtered_df` is used to call the user's `score` method, so as
    # part of return value we only send the columns that were present in
    # the original input dataframe.  Moderation information should not be
    # the filtered_df
    return prescore_df, filtered_df[data.columns], prescore_latency


def run_postscore_guards(pipeline, predictions_df):
    """
    Run postscore guards on the input data.

    Args:
        pipeline: Guard Pipeline
        predictions_df: This is the dataframe obtained as a return value from
            user's `score` method

    Returns:
        postscore_df: Dataframe with all moderations applied to the predictions_df.
            It has all the moderation information into various columns and is required
            to build the final result dataframe (as `postscore_df` argument to
            the method `format_result_df`)
        postscore_latency: Latency of executing postscore guards
    """
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    blocked_completion_column_name = f"blocked_{response_column_name}"
    input_df = predictions_df.copy(deep=True)
    if len(pipeline.get_postscore_guards()) == 0:
        input_df[blocked_completion_column_name] = False
        return input_df, 0

    start_time = time.time()
    try:
        postscore_df, postscore_latency = AsyncGuardExecutor(pipeline).run_guards(
            input_df, pipeline.get_postscore_guards(), GuardStage.RESPONSE
        )
    except Exception as ex:
        end_time = time.time()
        _logger.error(f"Failed to run postscore guards: {ex}")
        _logger.error(traceback.format_exc())
        postscore_df = input_df
        postscore_df[blocked_completion_column_name] = False
        postscore_latency = end_time - start_time

    # Again ensure the indexing matches the input dataframe indexing
    postscore_df.index = predictions_df.index
    _logger.debug(postscore_df)

    return postscore_df, postscore_latency


def run_user_score_function(filtered_df, model, pipeline, drum_score_fn, **kwargs):
    """
    A wrapper to execute user's `score` method.  Wrapper is useful to calculate the
    latency of the `score` method and handle any exceptional conditions

    Args:
        filtered_df: Input DataFrame to execute `score` on.  In the presence of
            prescore guards, it should be `filtered_df` returned by the method
            `run_prescore_guards`.  Otherwise, it is an input dataframe received
             from the user
        model: Model object as passed by DRUM
        pipeline: Guard Pipeline
        drum_score_fn: The `score` method to execute
        **kwargs:

    Returns:
        predictions_df: DataFrame obtained as a return value from user's `score`
            method
        score_latency: Latency to execute user's `score` method
    """
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    start_time = time.time()

    try:
        predictions_df = drum_score_fn(filtered_df, model, **kwargs)
    except Exception as e:
        _logger.error(f"Failed to execute user score function: {e}")
        pd.set_option("display.max_columns", None)
        _logger.error(filtered_df)
        raise

    if response_column_name not in predictions_df.columns:
        _logger.error("Missing response column in predictions df, can't run postscore guards")
        _logger.error(f"Columns received: {predictions_df.columns}")
        _logger.error(f"Response column expected: {response_column_name}")
        pd.set_option("display.max_columns", None)
        _logger.error(predictions_df)
        raise Exception(
            f"Response column name {response_column_name} is missing in "
            "the predictions df returned by custom.py"
        )

    # Because 'score' function index is not same as filtered data index
    # we need to match the indexes first
    predictions_df.index = filtered_df.index
    end_time = time.time()
    score_latency = end_time - start_time
    pipeline.report_score_latency(score_latency)
    return predictions_df, score_latency

def run_user_chat_function(completion_create_params, model, pipeline, drum_chat_fn, **kwargs):
    """
    A wrapper to execute user's `score` method.  Wrapper is useful to calculate the
    latency of the `score` method and handle any exceptional conditions

    Args:
        filtered_df: Input DataFrame to execute `score` on.  In the presence of
            prescore guards, it should be `filtered_df` returned by the method
            `run_prescore_guards`.  Otherwise, it is an input dataframe received
             from the user
        model: Model object as passed by DRUM
        pipeline: Guard Pipeline
        drum_chat_fn: The `score` method to execute
        **kwargs:

    Returns:
        predictions_df: DataFrame obtained as a return value from user's `score`
            method
        score_latency: Latency to execute user's `score` method
    """
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    start_time = time.time()

    try:
        response = drum_chat_fn(model, completion_create_params, **kwargs)
    except Exception as e:
        _logger.error(f"Failed to execute user chat function: {e}")
        pd.set_option("display.max_columns", None)
        raise

    end_time = time.time()
    score_latency = end_time - start_time
    pipeline.report_score_latency(score_latency)
    return response, score_latency


def guard_score_wrapper(data, model, pipeline, drum_score_fn, **kwargs):
    """
    Score wrapper function provided by the moderation library.  DRUM will invoke this
    function with the user's score function.  The wrapper will execute following steps:

        1.  Run prescore guards
        2.  Execute user's `score` method
        3.  Run postscore guards
        4.  Assemble the result dataframe using output from steps 1 to 3
        5.  Perform additional metadata calculations (eg. token counts, confidence
            score etc)

    Args:
        data: Input dataframe sent for predictions by the user
        model: Model object as passed by DRUM
        pipeline: Guard Pipeline (initialized in the `init()` call
        drum_score_fn: User's `score` method
    :return:
    """
    _logger.debug(data)

    pipeline.get_new_metrics_payload()
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)

    # ==================================================================
    # Step 1: Prescore Guards processing
    #
    prescore_df, filtered_df, prescore_latency = run_prescore_guards(pipeline, data)

    _logger.debug("After passing input through pre score guards")
    _logger.debug(filtered_df)
    _logger.debug(f"Pre Score Guard Latency: {prescore_latency} sec")

    if filtered_df.empty:
        blocked_message_prompt_column_name = f"blocked_message_{prompt_column_name}"
        # If all prompts in the input are blocked, means no need to
        # run score function and postscore guards, just simply return
        # the prescore_df
        prescore_df.rename(
            columns={
                blocked_message_prompt_column_name: pipeline.get_input_column(GuardStage.RESPONSE)
            },
            inplace=True,
        )
        pipeline.report_custom_metrics(prescore_df)
        return _handle_result_df_error_cases(prompt_column_name, prescore_df, prescore_latency)
    # ==================================================================

    # ==================================================================
    # Step 2: custom.py `score` call
    #
    predictions_df, score_latency = run_user_score_function(
        filtered_df, model, pipeline, drum_score_fn, **kwargs
    )
    _logger.debug("After invoking user's score function")
    _logger.debug(predictions_df)

    # Don't lose the association ids if they exist:
    association_id_column_name = pipeline.get_association_id_column_name()
    if (
        association_id_column_name
        and association_id_column_name not in predictions_df.columns
        and association_id_column_name in filtered_df.columns
    ):
        predictions_df[association_id_column_name] = filtered_df[association_id_column_name]
    # ==================================================================

    # ==================================================================
    # Step 3: Postscore Guards processing
    #
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    # Required for faithfulness calculation, we get prompt from the filtered_df
    # because it will use the replaced prompt if present.
    predictions_df[prompt_column_name] = filtered_df[prompt_column_name]

    postscore_df, postscore_latency = run_postscore_guards(pipeline, predictions_df)
    _logger.debug("After passing completions through post score guards")
    _logger.debug(postscore_df)
    _logger.debug(f"Post Score Guard Latency: {postscore_latency} sec")

    # ==================================================================
    # Step 4: Assemble the result - we need to merge prescore, postscore
    #         Dataframes.
    #
    result_df = format_result_df(pipeline, prescore_df, postscore_df, data)

    # ==================================================================
    # Step 5: Additional metadata calculations
    #
    result_df["datarobot_latency"] = (
        score_latency + prescore_latency + postscore_latency
    ) / result_df.shape[0]

    calculate_token_counts_and_confidence_score(pipeline, result_df)

    return result_df

def guard_chat_wrapper(completion_create_params, model, pipeline, drum_chat_fn, **kwargs):
    """
    Score wrapper function provided by the moderation library.  DRUM will invoke this
    function with the user's score function.  The wrapper will execute following steps:

        1.  Run prescore guards
        2.  Execute user's `score` method
        3.  Run postscore guards
        4.  Assemble the result dataframe using output from steps 1 to 3
        5.  Perform additional metadata calculations (eg. token counts, confidence
            score etc)

    Args:
        data: Input dataframe sent for predictions by the user
        model: Model object as passed by DRUM
        pipeline: Guard Pipeline (initialized in the `init()` call
        drum_score_fn: User's `score` method
    :return:
    """

    pipeline.get_new_metrics_payload()

    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)

    latest_message = completion_create_params["messages"][-1]["content"]
    data = pd.DataFrame([{prompt_column_name: latest_message,
                          pipeline.get_association_id_column_name(): str(uuid.uuid4())}])

    _logger.debug(data)



    # ==================================================================
    # Step 1: Prescore Guards processing
    #
    prescore_df, filtered_df, prescore_latency = run_prescore_guards(pipeline, data)

    _logger.debug("After passing input through pre score guards")
    _logger.debug(filtered_df)
    _logger.debug(f"Pre Score Guard Latency: {prescore_latency} sec")

    if filtered_df.empty:
        blocked_message_prompt_column_name = f"blocked_message_{prompt_column_name}"
        # If all prompts in the input are blocked, means no need to
        # run score function and postscore guards, just simply return
        # the prescore_df
        prescore_df.rename(
            columns={
                blocked_message_prompt_column_name: pipeline.get_input_column(GuardStage.RESPONSE)
            },
            inplace=True,
        )
        _logger.debug(prescore_df)

        pipeline.report_custom_metrics(prescore_df)
        return _handle_result_df_error_cases(prompt_column_name, prescore_df, prescore_latency)
    # ==================================================================

    # ==================================================================
    # Step 2: custom.py `score` call
    #
    response, score_latency = run_user_chat_function(
        completion_create_params, model, pipeline, drum_chat_fn, **kwargs
    )
    _logger.debug("After invoking user's chat function")
    _logger.debug(response)

    predictions_df = pd.DataFrame([{"response": response.choices[0].message.content}])

    # Don't lose the association ids if they exist:
    association_id_column_name = pipeline.get_association_id_column_name()
    if (
        association_id_column_name
        and association_id_column_name not in predictions_df.columns
        and association_id_column_name in filtered_df.columns
    ):
        predictions_df[association_id_column_name] = filtered_df[association_id_column_name]
    # ==================================================================

    # ==================================================================
    # Step 3: Postscore Guards processing
    #

    # Required for faithfulness calculation, we get prompt from the filtered_df
    # because it will use the replaced prompt if present.
    predictions_df[prompt_column_name] = filtered_df[prompt_column_name]

    postscore_df, postscore_latency = run_postscore_guards(pipeline, predictions_df)
    _logger.debug("After passing completions through post score guards")
    _logger.debug(postscore_df)
    _logger.debug(f"Post Score Guard Latency: {postscore_latency} sec")

    # ==================================================================
    # Step 4: Assemble the result - we need to merge prescore, postscore
    #         Dataframes.
    #
    result_df = format_result_df(pipeline, prescore_df, postscore_df, data)

    # ==================================================================
    # Step 5: Additional metadata calculations
    #
    result_df["datarobot_latency"] = (
        score_latency + prescore_latency + postscore_latency
    ) / result_df.shape[0]

    calculate_token_counts_and_confidence_score(pipeline, result_df)

    return response


def format_result_df(pipeline, prescore_df, postscore_df, data):
    """
    Build the final response dataframe to be returned as response using
    moderation information from prescore and postscore guards as well as
    input dataframe

    Args:
        pipeline: Guard Pipeline
        prescore_df: `prescore_df` obtained from `run_prescore_guards`
        postscore_df: `postscore_df` obtained from `run_postscore_guards`
        data: Input dataframe sent for predictions by the user

    Returns:
        result_df: Final dataframe with predictions and moderation information
            combined to be returned to the user

    """
    prompt_column_name = pipeline.get_input_column(GuardStage.PROMPT)
    blocked_prompt_column_name = f"blocked_{prompt_column_name}"
    blocked_message_prompt_column_name = f"blocked_message_{prompt_column_name}"
    response_column_name = pipeline.get_input_column(GuardStage.RESPONSE)
    blocked_completion_column_name = f"blocked_{response_column_name}"

    # This is the final result_df to be returned to the user
    result_columns = (
        set(postscore_df.columns)
        .union(set(prescore_df.columns))
        .union(set(datarobot_metadata_columns))
    )
    result_df = pd.DataFrame(index=data.index, columns=list(result_columns))

    # for the blocked prompts, their completion is the blocked message
    # configured by the guard
    for index, row in prescore_df.iterrows():
        if row.get(blocked_prompt_column_name):
            result_df.loc[index, response_column_name] = row[blocked_message_prompt_column_name]
        # Copy metric columns from prescore_df - it has prediction values from
        # the prescore guards, whether prescore guard blocked the text or not
        # what action prescore guard took on that prompt etc
        for column in prescore_df.columns:
            result_df.loc[index, column] = row[column]

    blocked_message_completion_column_name = f"blocked_message_{response_column_name}"
    replaced_message_prompt_column_name = f"replaced_message_{prompt_column_name}"
    replaced_response_column_name = f"replaced_{response_column_name}"
    replaced_message_response_column_name = f"replaced_message_{response_column_name}"
    # Now for the rest of the prompts, we did get completions.  If the completion
    # is blocked, use that message, else use the completion.  Note that, even if
    # PII Guard has replaced the completion, it will still be under row['completion']
    for index, row in postscore_df.iterrows():
        if row.get(blocked_completion_column_name):
            result_df.loc[index, response_column_name] = row[blocked_message_completion_column_name]
        elif row.get(replaced_response_column_name):
            result_df.loc[index, response_column_name] = row[replaced_message_response_column_name]
        else:
            result_df.loc[index, response_column_name] = row[response_column_name]
        # Similarly, copy metric columns from the postscore df - it has prediction
        # values from the postscore guards, whether postscore guard blocked the
        # completion or reported the completion, what action postscore guard took on
        # that completion, citations etc
        for column in postscore_df.columns:
            if column != response_column_name:
                result_df.loc[index, column] = row[column]

    # We don't need these 2 columns, because they have already been copied into
    # 'completion' column
    for column in [
        blocked_message_prompt_column_name,
        blocked_message_completion_column_name,
        replaced_message_prompt_column_name,
        replaced_message_response_column_name,
        f"Noneed_{prompt_column_name}",
        f"Noneed_{response_column_name}",
    ]:
        if column in result_df.columns:
            result_df = result_df.drop(column, axis=1)

    block_citations_if_prompt_blocked(pipeline, result_df)

    # Single call custom metric reporting
    pipeline.report_custom_metrics(result_df)

    # Also, ensure that result_df does not contain columns from the input df, creates problem
    # during the data export
    for column in data.columns:
        if column in result_df.columns:
            result_df.drop(column, axis=1, inplace=True)

    _logger.debug("Return df")
    _logger.debug(result_df)

    return result_df


def init():
    """
    Initialize the moderation framework

    Returns:
        pipeline: A Guard pipeline object required to enforce moderations while
            scoring on user data
    """
    guard_config_file = DEFAULT_GUARD_CONFIG_FILE
    if not os.path.exists(guard_config_file):
        _logger.warning(
            f"Guard config file: {guard_config_file} not found in the model directory,"
            " moderations will not be enforced on this model"
        )
        return None
    pipeline = Pipeline(guard_config_file)
    model_dir = os.getcwd()
    pipeline.set_model_dir(model_dir)
    # Lets export the PROMPT_COLUMN_NAME for custom.py
    os.environ["PROMPT_COLUMN_NAME"] = pipeline.get_input_column(GuardStage.PROMPT)
    os.environ["RESPONSE_COLUMN_NAME"] = pipeline.get_input_column(GuardStage.RESPONSE)
    return pipeline
