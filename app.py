# Import necessary modules
from flask import Flask, render_template, request, Response
from model import PredictionPipeline

# Create a Flask web application
app = Flask(__name__)

# Initialize a PredictionPipeline object
pipeline = PredictionPipeline()
pipeline.load_model_and_tokenizers()
pipeline.load_sentence_transformer()
pipeline.load_reranking_model()
pipeline.load_embeddings()

# Define a route for the root URL ('/'), rendering an HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for the '/stream' URL
@app.route('/stream')
def stream():
    # Retrieve input text from query parameters
    text = request.args.get('text')

    # Return predictions in a streaming fashion using Server-Sent Events (SSE)
    return Response(pipeline.make_predictions(text), content_type='text/event-stream')

if __name__ == '__main__':
    # Listen on all available network interfaces ('0.0.0.0') on port 8000
    app.run(host='0.0.0.0', port=8000, debug=False)
