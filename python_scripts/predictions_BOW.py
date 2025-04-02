#!/usr/bin/env python3
# Script for predicting deception from new messages using saved models
import argparse
import pickle
import spacy
from spacy.lang.en import English
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os

def parse_arguments():
    """Parse command line arguments for prediction script."""
    parser = argparse.ArgumentParser(description='Predict deception in messages using trained models.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model pickle file')
    parser.add_argument('--vectorizer_path', type=str, required=True, help='Path to the saved vectorizer pickle file')
    parser.add_argument('--message', type=str, help='Message to analyze')
    parser.add_argument('--message_file', type=str, help='File containing messages to analyze, one per line')
    parser.add_argument('--use_power', action='store_true', help='Use power features in prediction')
    parser.add_argument('--power_delta', type=int, default=0, help='Power delta to use for prediction')
    parser.add_argument('--power_threshold', type=int, default=4, help='Threshold for power score delta')
    return parser.parse_args()


def check_if_number(token):
    """Check if the given token is a number."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def spacy_tokenizer(text, tokenizer_model):
    """Tokenize the input text using the spaCy tokenizer."""
    tokenized_text = tokenizer_model(text)
    return [token.text if not check_if_number(token.text) else '_NUM_' for token in tokenized_text]


def load_model(model_path, vectorizer_path):
    """Load model and vectorizer from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        print(f"Check if the paths are correct and the files exist:")
        print(f"Model path: {model_path}")
        print(f"Vectorizer path: {vectorizer_path}")
        return None, None


def predict_lie(message, model, vectorizer, tokenizer_model, use_power=True, power_delta=0, power_threshold=4, verbose=False):
    """Predict whether a message contains a lie using the trained model."""
    if verbose:
        print(f"\nPredicting deception for message: \"{message}\"")
        print(f"Using power features: {use_power}, Power delta: {power_delta}, Threshold: {power_threshold}")
    """Predict whether a message contains a lie using the trained model."""
    # Pre-tokenize the message
    tokens = spacy_tokenizer(message.lower(), tokenizer_model)
    tokenized_message = ' '.join(tokens)
    
    if verbose:
        print(f"Tokenized message: {tokenized_message}")
    
    # Create a vectorizer with the saved vocabulary
    new_vectorizer = CountVectorizer(
        vocabulary=vectorizer.vocabulary_,
        stop_words='english',
        strip_accents='unicode'
    )
    
    # Vectorizing the pre-tokenized message
    message_features = new_vectorizer.transform([tokenized_message])
    
    if verbose:
        print(f"Message vectorized with {message_features.shape[1]} features")
    
    # Adding power features to the total feature matrix for the message if requested
    if use_power:
        # We always need to add power features for models trained with power
        power_features = []
        if power_delta > power_threshold:
            power_features.append(1)
        else:
            power_features.append(0)
        if power_delta < -power_threshold:
            power_features.append(1)
        else:
            power_features.append(0)
        
        if verbose:
            print(f"Power delta: {power_delta} (high: {power_delta > power_threshold}, low: {power_delta < -power_threshold})")
            
        # Combining the power features with the count vectorizer's features
        combined_features = np.append(message_features.toarray(), [power_features], axis=1)
        message_features = csr_matrix(combined_features)
        
        if verbose:
            print(f"Message features with power: {message_features.shape[1]} features")
    
    # Making the predictions using the trained model
    prediction = model.predict(message_features)[0]
    probabilities = model.predict_proba(message_features)[0]
    
    # Returning the prediction results for the given message
    return {
        'is_lie': prediction == 0,  # 0 means lie
        'confidence': probabilities[0] if prediction == 0 else probabilities[1], 
        'prediction': 'Lie' if prediction == 0 else 'Truth',
        'probabilities': {
            'lie': probabilities[0],
            'truth': probabilities[1]
        }
    }


def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print script execution info
    print("*" * 60)
    print("Deception Detection Prediction Script")
    print("*" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Vectorizer path: {args.vectorizer_path}")
    
    # Check if paths exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist at {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.vectorizer_path):
        print(f"Error: Vectorizer file does not exist at {args.vectorizer_path}")
        sys.exit(1)
    
    # Initialize spaCy
    try:
        tokenizer_model = English()
        print("SpaCy English model loaded successfully.")
    except Exception as e:
        print(f"Error loading spaCy English model: {e}")
        print("Try installing with: python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    # Load the model and vectorizer
    model, vectorizer = load_model(args.model_path, args.vectorizer_path)
    if model is None or vectorizer is None:
        sys.exit(1)
    
    print(f"Model and vectorizer loaded successfully")
    
    # Process messages
    messages = []
    if args.message:
        messages.append(args.message)
        print(f"Analyzing message: \"{args.message}\"")
    elif args.message_file:
        try:
            with open(args.message_file, 'r') as f:
                messages = [line.strip() for line in f if line.strip()]
            print(f"Analyzing {len(messages)} messages from file: {args.message_file}")
        except Exception as e:
            print(f"Error reading message file: {e}")
            sys.exit(1)
    else:
        print("Please provide either --message or --message_file")
        sys.exit(1)
        
    # Check if we need to force use_power based on model name
    if "with_power" in args.model_path and not args.use_power:
        print("WARNING: Model appears to be trained with power features but --use_power flag is not set.")
        print("Automatically enabling power features for prediction.")
        args.use_power = True
    
    # Analyze messages
    print(f"\nAnalysis Results:")
    print("-" * 60)
    
    for i, message in enumerate(messages):
        result = predict_lie(
            message, 
            model, 
            vectorizer, 
            tokenizer_model, 
            use_power=args.use_power, 
            power_delta=args.power_delta,
            power_threshold=args.power_threshold,
            verbose=True
        )
        
        print(f"\nMessage {i+1}: \"{message}\"")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
        print(f"Probabilities: Lie: {result['probabilities']['lie']:.2f}, Truth: {result['probabilities']['truth']:.2f}")
        print("-" * 60)
    

if __name__ == "__main__":
    main()