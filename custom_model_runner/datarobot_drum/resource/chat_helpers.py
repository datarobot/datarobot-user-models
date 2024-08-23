def is_streaming_response(response):
    return getattr(response, "object", None) != "chat.completion"
