from flask import Flask, request, jsonify
from model import KeywordExtractor
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize the keyword extractor model
extractor = KeywordExtractor()


@app.route('/extract', methods=['POST'])
def extract_keywords():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({'error': 'Please provide a title in the JSON payload.'}), 400

    title = data['title']
    keywords = extractor.extract(title)
    return jsonify({'keywords': keywords}), 200


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    print(extractor.extract("This is a test title"))
