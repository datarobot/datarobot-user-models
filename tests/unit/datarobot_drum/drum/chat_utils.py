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


def create_completion_chunks(messages):
    def create_chunk(content, finish_reason=None, role=None):
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
    #  OpenAI returns a chunk with empty string in beginning of stream
    chunks.append(create_chunk("", role="assistant"))

    for message in messages:
        chunks.append(create_chunk(message))

    #  And a None chunk in the end
    chunks.append(create_chunk(None, finish_reason="stop"))
    return chunks
