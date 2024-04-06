def stream_predict(query, history):
    history.append((query, ''))

    prompt = ''
    if len(history) > 1:
        prompt += "{}\n\n答：{}\n\n".format(history[0][0], history[0][1])
        for i, (old_query, response) in enumerate(history[1:-1]):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 2, old_query, response)
        prompt += "[Round {}]\n\n问：{}".format(len(history), query)
    else:
        prompt += "{}".format(query)

    return prompt

print(stream_predict("今天天气怎么样", [("你好", "你好，我是人工智能助手"), ("你好", "你好，我是人工智能助手"), ("你好", "你好，我是人工智能助手")]))
