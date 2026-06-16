#!/bin/bash
set -e

PYTHON="python3.10"
VENV_DIR="venv"

if ! command -v $PYTHON &>/dev/null; then
    echo "python3.10 not found. Install it via: brew install python@3.10"
    exit 1
fi

echo "Creating venv with $($PYTHON --version)..."
$PYTHON -m venv $VENV_DIR

echo "Activating and upgrading pip..."
source $VENV_DIR/bin/activate
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

# scispaCy model — small general biomedical model for NER + dependency parsing
echo "Installing scispaCy model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# spaCy English model for general sentence parsing
echo "Installing spaCy English model..."
python -m spacy download en_core_web_sm

echo "Registering Jupyter kernel as 'nlp-env'..."
python -m ipykernel install --user --name nlp-env --display-name "Python (nlp-env)"

echo ""
echo "Done. Activate with: source venv/bin/activate"
