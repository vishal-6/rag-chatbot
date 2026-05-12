from flask import Flask, request, jsonify
from rag_pipeline import get_qa_chain

app = Flask(__name__)
qa_chain = get_qa_chain()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = qa_chain(question)
    return jsonify({"answer": result["result"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)