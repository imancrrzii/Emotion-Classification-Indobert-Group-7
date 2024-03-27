from flask import Flask, request, render_template, jsonify
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Mapping model and tokenizer paths based on dropdown selections
model_tokenizer_mapping = {
    'model1': {
        'model_checkpoint_path': 'model/model-original/model-bert.pt',
        'tokenizer_directory': 'model/model-original/'
    },
    'model2': {
        'model_checkpoint_path': 'model/model-finetuned-79-77-78-77/fine_tuned_model_checkpoint3.pt',
        'tokenizer_directory': 'model/model-finetuned-79-77-78-77/'
    },
    'model3': {
        'model_checkpoint_path': 'model/model-finetuned-81-78-78-79/fine_tuned_model_checkpoint10.pt',
        'tokenizer_directory': 'model/model-finetuned-81-78-78-79/'
    }
}

default_model = 'model1'
default_tokenizer = 'model1'

model_checkpoint_path = model_tokenizer_mapping[default_model]['model_checkpoint_path']
tokenizer_directory = model_tokenizer_mapping[default_tokenizer]['tokenizer_directory']

checkpoint = torch.load(model_checkpoint_path,
                        map_location=torch.device('cpu'))
tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
config = checkpoint['config']
model = BertForSequenceClassification(config)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

label_mapping = {
    0: 'sedih',
    1: 'marah',
    2: 'cinta',
    3: 'takut',
    4: 'senang',
    5: 'jijik',
}


def predict_label(text, tokenizer, model):
    subwords = tokenizer.encode(text, add_special_tokens=True)
    subwords = torch.LongTensor(subwords).view(1, -1)

    with torch.no_grad():
        logits = model(subwords)[0]

    label_id = torch.argmax(logits).item()
    label = label_mapping.get(label_id, 'unknown')
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()

    return label, probabilities


@app.route('/', methods=['GET', 'POST'])
def home():
    label = None
    confidence = None
    if request.method == 'POST':
        text = request.form['sentence']
        model_selection = request.form['model']
        tokenizer_selection = request.form['tokenizer']

        model_checkpoint_path = model_tokenizer_mapping[model_selection]['model_checkpoint_path']
        tokenizer_directory = model_tokenizer_mapping[tokenizer_selection]['tokenizer_directory']

        checkpoint = torch.load(model_checkpoint_path,
                                map_location=torch.device('cpu'))
        tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
        config = checkpoint['config']
        model = BertForSequenceClassification(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        label, confidence = predict_label(text, tokenizer, model)

    return render_template('index.html', label=label, confidence=confidence)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['sentence']
    model_selection = data['model']
    tokenizer_selection = data['tokenizer']

    model_checkpoint_path = model_tokenizer_mapping[model_selection]['model_checkpoint_path']
    tokenizer_directory = model_tokenizer_mapping[tokenizer_selection]['tokenizer_directory']

    checkpoint = torch.load(model_checkpoint_path,
                            map_location=torch.device('cpu'))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_directory)
    config = checkpoint['config']
    model = BertForSequenceClassification(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    label, probabilities = predict_label(text, tokenizer, model)

    response = {
        "emotion": label,
        "probabilities": {label_mapping[i]: float(probabilities[i]) for i in range(len(probabilities))}
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
