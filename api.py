from flask import Flask, request, jsonify
from chat import DocChatbot

app = Flask(__name__)
chatbot_st = DocChatbot.get_instance()

# if you have local db, load it
chatbot_st.load_first_vector_db()

app.config['JSON_AS_ASCII'] = False

# export LLM_MODEL="qwen"

@app.route("/chatdoc", methods=['POST'])
def chatdoc():
    data = request.get_json()
    question = data['question']
    docs = chatbot_st.query_from_doc(question, 3)
    refer = "\n".join([x.page_content.replace("\n", '\t') for x in docs])
    PROMPT = """{}。\n请根据下面的参考文档回答上述问题。\n{}\n"""
    prompt = PROMPT.format(question, refer)

    res = ''
    for result_answer, _ in chatbot_st.llm.stream_predict(prompt, []):
        res = result_answer

    response = {
        'result': res
    }
    return jsonify(response)


@app.route("/chatbot", methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data['question']
    history = data['history']
    history = [(x[0], x[1]) for x in history]
    res = ''
    for result_answer, _ in chatbot_st.llm.stream_predict(question, history):
        res = result_answer

    response = {
        'result': res
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0')