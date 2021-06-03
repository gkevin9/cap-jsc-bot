import flask
from flask import request, jsonify
from inference import main

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "Hello flask"

@app.route('/api/v1/question/', methods=['GET'])
def api_question():
    if 'question' in request.args:
        question = request.args['question']
    else:
        return "Error: Question not found"

    if question != "":
        tempAnswer = main(question)
        answer = {"answer": tempAnswer}
    else:
        answer = {"answer": "no"}
        
    return jsonify(answer)
    # answer = main(question)
    # return answer

app.run(host="0.0.0.0", port=5000)