from openai.types.chat import ChatCompletionMessage, ChatCompletionChunk, chat_completion_chunk
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_chunk import ChoiceDelta


def create_completion(message_content):
    return ChatCompletion(
        id="id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=message_content),
            )
        ],
        created=123,
        model="model",
        object="chat.completion",
    )


def create_completion_chunks(messages, use_custom_streaming_class=False):
    class CustomModelStreamingResponse(ChatCompletionChunk):
        pass

    def create_chunk(content, finish_reason=None, role=None):
        if use_custom_streaming_class:
            return CustomModelStreamingResponse(
                id="id",
                choices=[
                    chat_completion_chunk.Choice(
                        delta=ChoiceDelta(content=content, role=role),
                        finish_reason=finish_reason,
                        index=0,
                    )
                ],
                created=0,
                model="model",
                object="chat.completion.chunk",
                pipeline_interactions="pipeline.interactions",
            )
        else:
            return ChatCompletionChunk(
                id="id",
                choices=[
                    chat_completion_chunk.Choice(
                        delta=ChoiceDelta(content=content, role=role),
                        finish_reason=finish_reason,
                        index=0,
                    )
                ],
                created=0,
                model="model",
                object="chat.completion.chunk",
            )

    chunks = []
    #  OpenAI returns a chunk with empty string and empty object in beginning of stream
    chunk = create_chunk("", role="assistant")
    if not use_custom_streaming_class:
        chunk.object = ""
    chunks.append(chunk)

    for message in messages:
        chunks.append(create_chunk(message))

    #  And a None chunk in the end
    chunks.append(create_chunk(None, finish_reason="stop"))
    return chunks
