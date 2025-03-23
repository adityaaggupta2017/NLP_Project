#!/usr/bin/env python3
# Importing the required libraries
import jsonlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import spacy
from spacy.lang.en import English
import warnings
import argparse
import os
import pickle
warnings.filterwarnings("ignore")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate deception detection models.')
    parser.add_argument('--data_path', type=str, default='Data/', help='Path to the dataset directory')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for LogisticRegression')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the trained models')
    parser.add_argument('--power_threshold', type=int, default=4, help='Threshold for power score delta')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--no_plots', action='store_true', help='Disable plotting')
    return parser.parse_args()


# Initialize spaCy for tokenization
def initialize_spacy():
    try:
        spacy_tokenizer = English()
        print("SpaCy English model loaded successfully.")
        return spacy_tokenizer
    except Exception as e:
        print(f"Error loading spaCy English model: {e}")
        print("Try installing with: python -m spacy download en_core_web_sm")
        return None


# Checking to see if the given token is a number or not
def check_if_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


# Defining the function to tokenize the input text using the Spacy tokenizer
def spacy_tokenizer(text, tokenizer_model):
    tokenized_text = tokenizer_model(text)
    # Return a list of tokenized words, replacing numbers with _NUM_
    return [token.text if not check_if_number(token.text) else '_NUM_' for token in tokenized_text]


# Function to aggregate data from dialogues into individual messages
def collate_dialogues(dataset):

    # Initializing lists to store the messages, receiver labels, sender labels and power scores for each message in each dialogue
    messages = []
    receiver_decption_labels = []
    sender_deception_labels = []
    power_score_deltas = []

    # Iterating over all the dialogues in the dataset
    for dialog_messages in dataset:

        # Extracting and collating the messages, receiver labels, sender labels, and power scores for each dialogue
        messages.extend(dialog_messages['messages'])
        receiver_decption_labels.extend(dialog_messages['receiver_labels'])
        sender_deception_labels.extend(dialog_messages['sender_labels'])
        power_score_deltas.extend(dialog_messages['game_score_delta'])
    
    # Initializing an empty dictionary to store the collated messages
    collated_dictionary = []

    # Iterating over the collated data points
    for i, item in enumerate(messages):

        # Creating a new dicitionary entry for each data point in the collated lists
        collated_dictionary.append({
            'message': item, 
            'sender_annotation': sender_deception_labels[i], 
            'receiver_annotation': receiver_decption_labels[i], 
            'score_delta': int(power_score_deltas[i])
        })
    
    # Returning the dictionary containing all the collated data points
    return collated_dictionary


# Converting the deception lables to binary labels
def convert_to_binary(dataset, task="SENDER", use_power=True, power_threshold=4):

    # Initializing the list to store the binary deception label data for all messages
    total_binary_data = []
    
    for message in dataset:
        # Skipping unannotated instances for RECEIVER task
        if message['receiver_annotation'] != True and message['receiver_annotation'] != False:
            if task == "SENDER":
                pass  # Keeping all for the SENDER task
            elif task == "RECEIVER":
                continue  # Skipping for the RECEIVER task
        
        # Initializing the list to store the binary deception label data for the current message
        cur_binary_data = []
        
        # Add power features if enabled (severe power skew)
        if use_power:
            # Strong positive power delta
            if message['score_delta'] > power_threshold:
                cur_binary_data.append(1)
            else:
                cur_binary_data.append(0)
            
            # Strong negative power delta
            if message['score_delta'] < -power_threshold:
                cur_binary_data.append(1)
            else:
                cur_binary_data.append(0)

        # Getting the deception label based on the given task
        if task == "SENDER":
            annotation = 'sender_annotation'
        elif task == "RECEIVER":
            annotation = 'receiver_annotation'
            
        # Adding the deception label to the current message's binary data
        if message[annotation] == False:
            cur_binary_data.append(0)  # 0 for False (lie)
        else:
            cur_binary_data.append(1)  # 1 for True (truth)

        total_binary_data.append(cur_binary_data)
    return total_binary_data


# Given the dataset, splitting the dataset into features (X) and labels (y)
def split_xy(data):

    # Initializing lists to store the features and labels
    X, Y = [], []

    # Iterating over each line in the given dataset
    for data_point in data:

        # Extracting the features from the current line of the dataset
        features = data_point[:len(data_point)-1]

        # Extracting the label from the current line of the dataset
        label = data_point[len(data_point)-1]

        # Adding the extracted features and label to the respective lists
        X.append(features)
        Y.append(label)

    # Returning the final lists of extracted features and labels
    return (X, Y)


# Defining the function to handle the model training and evaluation tasks for the given train and test datasets
def train_and_evaluate(training_dataset, testing_dataset, task="SENDER", use_power=True, 
                       max_iter=1000, random_state=42, power_threshold=4, plot=True, tokenizer_model=None):
    print(f"\n{'='*60}")
    print(f"Training model for {task} task with power={use_power}")
    print(f"{'='*60}\n")
    
    # Collating the messages for all dialogues in the given training dataset to create a new collated training dataset
    training_collated_dataset = collate_dialogues(training_dataset)

    # Collating the messages for all dialogues in the given testing dataset to create a new collated testing dataset
    testing_collated_dataset = collate_dialogues(testing_dataset)
    
    # Preparing the training dataset
    if task == "SENDER":
        corpus = [message['message'].lower() for message in training_collated_dataset]
    elif task == "RECEIVER": # For receivers, dropping all missing annotations        
        corpus = [message['message'].lower() for message in training_collated_dataset if message['receiver_annotation'] == True or message['receiver_annotation'] == False]
    
    # Tokenize the corpus first
    tokenized_corpus = []
    for text in corpus:
        tokens = spacy_tokenizer(text, tokenizer_model)
        tokenized_corpus.append(' '.join(tokens))
    
    # Creating the count vectorizer object without custom tokenizer function
    vectorizer = CountVectorizer(
        stop_words='english',  # Using predefined English stopwords
        strip_accents='unicode'
    )
    
    # Transforming the tokenized corpus to BoW features
    training_features = vectorizer.fit_transform(tokenized_corpus)
 
    # Preparing the testing dataset
    if task == "SENDER":
        test_corpus = [message['message'].lower() for message in testing_collated_dataset]
    elif task == "RECEIVER": # For receivers, drop all missing annotations        
        test_corpus = [message['message'].lower() for message in testing_collated_dataset if message['receiver_annotation'] == True or message['receiver_annotation'] == False]
    
    # Tokenize the test corpus first
    tokenized_test_corpus = []
    for text in test_corpus:
        tokens = spacy_tokenizer(text, tokenizer_model)
        tokenized_test_corpus.append(' '.join(tokens))
    
    # Creating the count vectorizer object with the same vocabulary
    test_vectorizer = CountVectorizer(
        vocabulary=vectorizer.vocabulary_,  # Specifying the use of the same vocabulary as the training dataset's vectorizer
        stop_words='english',  # Using predefined English stopwords
        strip_accents='unicode'
    )
    
    # Transforming the tokenized test corpus to BoW features
    testing_features = test_vectorizer.transform(tokenized_test_corpus)

    # Getting the binary labels for the training and testing datasets
    training_binary_data = convert_to_binary(training_collated_dataset, task, use_power, power_threshold)
    testing_binary_data = convert_to_binary(testing_collated_dataset, task, use_power, power_threshold)
    
    # Splitting the training and testing datasets' binary labels into power features and labels
    train_split_features, train_split_labels = split_xy(training_binary_data)
    test_split_features, test_split_labels = split_xy(testing_binary_data)

    # Appending the power features to both the original training and testing features generated by the Count Vectorizer if the power is to be used for training the model
    if use_power:
        new_training_features = np.append(training_features.toarray(), train_split_features, axis=1)
        new_testing_features = np.append(testing_features.toarray(), test_split_features, axis=1)
        
        # Converting feature matrices for the training and testing datasets back to sparse format
        training_features = csr_matrix(new_training_features)
        testing_features = csr_matrix(new_testing_features)
    
    # Printing the shape of the training and testing final feature matrix
    print(f"Training features shape: {training_features.shape}")
    print(f"Testing features shape: {testing_features.shape}")

    # Printing the number of Truths and Lies in the binary label dataset for the training dataset
    print(f"Label distribution in training labels: Truth={sum(train_split_labels)}, Lie={len(train_split_labels)-sum(train_split_labels)}")
    
    # Training the logistic regression model with balanced class weights (to handle imbalanced datasets) using specified epochs
    logmodel = LogisticRegression(class_weight='balanced', max_iter=max_iter, random_state=random_state)
    logmodel.fit(training_features, train_split_labels)
    
    # Making the predictions for the testing dataset using the trained Logistic Regression Model
    predictions = logmodel.predict(testing_features)
    
    # Printing the classification report for the testing dataset
    print("\nClassification Report:")
    report = classification_report(test_split_labels, predictions, digits=3, output_dict=True)
    print(classification_report(test_split_labels, predictions, digits=3))
    
    # Analysing the feature importance
    if use_power:
        # Handle the case where power features are included
        feature_names = list(vectorizer.get_feature_names_out()) + ['power_high', 'power_low']
    else:
        feature_names = list(vectorizer.get_feature_names_out())
    
    # Getting the coefficients for each of the feature vectors from the trained LR model
    coef = logmodel.coef_[0]
    
    # Creating a DataFrame to store the feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coef
    })
    
    # Sorting the created dataframe in descending order by absolute feature importance
    feature_importance['abs_importance'] = feature_importance['importance'].abs()
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
    
    # Printing the top 10 features that indicate a lie from the sorted list of features according to absolute importance
    print("\nTop 10 features indicating a lie (negative coefficients):")
    lie_features = feature_importance[feature_importance['importance'] < 0].head(10)
    print(lie_features[['feature', 'importance']])
    
    # Printing the top 10 features that indicate a truth from the sorted list of features according to absolute importance
    print("\nTop 10 features indicating truth (positive coefficients):")
    truth_features = feature_importance[feature_importance['importance'] > 0].head(10)
    print(truth_features[['feature', 'importance']])
    
    # Plotting the most important features that help identify truths and lies
    if plot:
        plt.figure(figsize=(12, 8))
        top_features = pd.concat([lie_features.head(10), truth_features.head(10)])
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top Features by Importance ({task} Task, Power={use_power})')
        plt.axvline(x=0, color='black', linestyle='-')
        plt.tight_layout()
        plt.savefig(f"{task}_power{use_power}_features.png")
        plt.show()
    
    # Returning the classification report on the testing dataset, the trained LR model for the given task, the trained Count Vectorizer object and the feature importance matrix for further processing and debugging
    return report, logmodel, vectorizer, feature_importance


# Defining the function to evaluate any the truthfulness of any given message using any one of the above trained models
def predict_lie(message, model, vectorizer, task="SENDER", use_power=True, power_delta=0, power_threshold=4, tokenizer_model=None):
    """Predict whether a message contains a lie using the trained model"""
    # Pre-tokenize the message
    tokens = spacy_tokenizer(message.lower(), tokenizer_model)
    tokenized_message = ' '.join(tokens)
    
    # Create a vectorizer with the saved vocabulary
    new_vectorizer = CountVectorizer(
        vocabulary=vectorizer.vocabulary_,
        stop_words='english',
        strip_accents='unicode'
    )
    
    # Vectorizing the pre-tokenized message
    message_features = new_vectorizer.transform([tokenized_message])
    
    # Adding power features to the total feature matrix for the message if requested
    if use_power:
        power_features = []
        if power_delta > power_threshold:
            power_features.append(1)
        else:
            power_features.append(0)
        if power_delta < -power_threshold:
            power_features.append(1)
        else:
            power_features.append(0)
            
        # Combining the power features with the count vectorizer's features to get the final feature vector for the message and converting it back to a sparse format
        combined_features = np.append(message_features.toarray(), [power_features], axis=1)
        message_features = csr_matrix(combined_features)
    
    # Making the predictions using the trained model
    prediction = model.predict(message_features)[0]
    probabilities = model.predict_proba(message_features)[0]
    
    # Returning the prediction results for the given message
    return {
        'is_lie': prediction == 0,  # 0 means lie
        'confidence': probabilities[0] if prediction == 0 else probabilities[1], # Confidence of the trained model in the predicted deception label
        'prediction': 'Lie' if prediction == 0 else 'Truth', # Prediction Label
        'probabilities': { # Returning the probability of that message being a truth or a lie
            'lie': probabilities[0],
            'truth': probabilities[1]
        }
    }


def save_model(model, vectorizer, task, use_power, save_path):
    """Save model and vectorizer to disk"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    power_str = "with_power" if use_power else "no_power"
    model_filename = os.path.join(save_path, f"{task}_{power_str}_model.pkl")
    vectorizer_filename = os.path.join(save_path, f"{task}_{power_str}_vectorizer.pkl")
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_filename, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_filename}")
    print(f"Vectorizer saved to {vectorizer_filename}")
    
    return model_filename, vectorizer_filename


def load_model(model_path, vectorizer_path):
    """Load model and vectorizer from disk"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def main():
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Initialize spaCy
    tokenizer_model = initialize_spacy()
    if tokenizer_model is None:
        print("Failed to initialize spaCy. Exiting.")
        return
    
    try:
        # Opening the train.jsonl file and loading the training dataset
        with jsonlines.open(os.path.join(args.data_path, 'train.jsonl'), 'r') as f:
            training_dataset = list(f)

        print("Successfully loaded the training dataset!")
        print("Training dataset samples: "+str(len(training_dataset)))

        # Opening the test.jsonl file and loading the training dataset
        with jsonlines.open(os.path.join(args.data_path, 'test.jsonl'), 'r') as f:
            testing_dataset = list(f)
        
        print("Successfully loaded the testing dataset!")
        print("Testing dataset samples: "+str(len(testing_dataset)))
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure the data files (train.jsonl and test.jsonl) are in the '{args.data_path}' directory.")
        return

    # Printing a data point from the dataset for debugging and understanding the structure and content of each data point
    print("Selecting the first dialogue!")
    print(f"Number of messages in this dialogue: {len(training_dataset[0]['messages'])}")
    for i, message in enumerate(training_dataset[0]['messages'][:5]):
        print(f"\nMessage {i+1}:")
        print("-----------")
        print("Message Content:")
        print(message)
        print("-----------")
        print(f"Sender Deception Label: {training_dataset[0]['sender_labels'][i]}")
        print("-----------")
        print(f"Receiver Deception Label: {training_dataset[0]['receiver_labels'][i]}")
        print("-----------")
        print(f"Power score delta: {training_dataset[0]['game_score_delta'][i]}")
        print("--------------------------------------------------------------------------------------------")

    # Collating the data points (messages) across all dialogues in the dataset and creating a new dictionary for these collated messages
    aggregated_train = collate_dialogues(training_dataset)

    # Counting the sender deception labels
    n_sender_true = sum(1 for msg in aggregated_train if msg['sender_annotation'] == True)
    n_sender_false = sum(1 for msg in aggregated_train if msg['sender_annotation'] == False)
    n_sender_unlabelled = sum(1 for msg in aggregated_train if msg['sender_annotation'] != True and msg['sender_annotation'] != False)

    # Count receiver annotations
    n_receiver_true = sum(1 for msg in aggregated_train if msg['receiver_annotation'] == True)
    n_receiver_false = sum(1 for msg in aggregated_train if msg['receiver_annotation'] == False)
    n_receiver_unlabelled = sum(1 for msg in aggregated_train if msg['receiver_annotation'] != True and msg['receiver_annotation'] != False)

    # Printing the class distribution statistics for both sender and receiver deception labels
    print("Sender Deception Label Class Distribution:")
    print(f"Sender Truth Label Percentage: {n_sender_true} ({n_sender_true/(n_sender_true+n_sender_false)*100:.1f}%)")
    print(f"Sender Lie Label Percentage: {n_sender_false} ({n_sender_false/(n_sender_true+n_sender_false)*100:.1f}%)")
    print(f"No annotation: {n_sender_unlabelled}")
    print()
    print("Receiver Deception Label Class Distribution:")
    print(f"Receiver Truth Label Percentage: {n_receiver_true} ({n_receiver_true/(n_receiver_true+n_receiver_false)*100:.1f}%)")
    print(f"Receiver Lie Label Percentage: {n_receiver_false} ({n_receiver_false/(n_receiver_true+n_receiver_false)*100:.1f}%)")
    print(f"No annotation: {n_receiver_unlabelled}")

    # Visualizing the class distributions for the sender and receiver decpetion labels
    if not args.no_plots:
        plt.figure(figsize=(12, 5))

        # Plot for the sender's deception labels
        plt.subplot(1, 2, 1)
        sns.barplot(x=['Truth', 'Lie'], y=[n_sender_true, n_sender_false])
        plt.title('Sender Annotations')
        plt.ylabel('Count')

        # Plot for the receiver's deception labels
        plt.subplot(1, 2, 2)
        sns.barplot(x=['Truth', 'Lie'], y=[n_receiver_true, n_receiver_false])
        plt.title('Receiver Annotations')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_path, 'class_distribution.png'))
        plt.show()

    # SENDER task with power
    sender_report_power, sender_model_power, sender_vectorizer_power, sender_features_power = train_and_evaluate(
        training_dataset, testing_dataset, task="SENDER", use_power=True, 
        max_iter=args.max_iter, random_state=args.random_state, power_threshold=args.power_threshold,
        plot=not args.no_plots, tokenizer_model=tokenizer_model
    )

    # SENDER task without power
    sender_report_no_power, sender_model_no_power, sender_vectorizer_no_power, sender_features_no_power = train_and_evaluate(
        training_dataset, testing_dataset, task="SENDER", use_power=False, 
        max_iter=args.max_iter, random_state=args.random_state, power_threshold=args.power_threshold,
        plot=not args.no_plots, tokenizer_model=tokenizer_model
    )

    # RECEIVER task with power
    receiver_report_power, receiver_model_power, receiver_vectorizer_power, receiver_features_power = train_and_evaluate(
        training_dataset, testing_dataset, task="RECEIVER", use_power=True, 
        max_iter=args.max_iter, random_state=args.random_state, power_threshold=args.power_threshold,
        plot=not args.no_plots, tokenizer_model=tokenizer_model
    )

    # RECEIVER task without power
    receiver_report_no_power, receiver_model_no_power, receiver_vectorizer_no_power, receiver_features_no_power = train_and_evaluate(
        training_dataset, testing_dataset, task="RECEIVER", use_power=False, 
        max_iter=args.max_iter, random_state=args.random_state, power_threshold=args.power_threshold,
        plot=not args.no_plots, tokenizer_model=tokenizer_model
    )

    # Save all models
    save_model(sender_model_power, sender_vectorizer_power, "SENDER", True, args.save_path)
    save_model(sender_model_no_power, sender_vectorizer_no_power, "SENDER", False, args.save_path)
    save_model(receiver_model_power, receiver_vectorizer_power, "RECEIVER", True, args.save_path)
    save_model(receiver_model_no_power, receiver_vectorizer_no_power, "RECEIVER", False, args.save_path)

    # Comparing the F1 scores for different models
    # Initializing the list to store all the types of models we trained before
    models = ['Sender+Power', 'Sender', 'Receiver+Power', 'Receiver']

    # Initializing the list to store the macro average F1-Score for all four types of models we trained above
    f1_macro = [
        sender_report_power['macro avg']['f1-score'],
        sender_report_no_power['macro avg']['f1-score'],
        receiver_report_power['macro avg']['f1-score'],
        receiver_report_no_power['macro avg']['f1-score']
    ]

    # Initializing the list to store the F1-Score for the Lie labels for all four types of models we trained above
    f1_lie = [
        sender_report_power['0']['f1-score'],  # 0 is lie (False) in our case
        sender_report_no_power['0']['f1-score'],
        receiver_report_power['0']['f1-score'],
        receiver_report_no_power['0']['f1-score']
    ]

    # Initializing the list to store the F1-Score for the Truth labels for all four types of models we trained above
    f1_truth = [
        sender_report_power['1']['f1-score'],  # 1 is truth (True) in our case
        sender_report_no_power['1']['f1-score'],
        receiver_report_power['1']['f1-score'],
        receiver_report_no_power['1']['f1-score']
    ]

    # Creating a DataFrame for comparison by consolidating all the data points for the four models
    comparison_df = pd.DataFrame({
        'Model': models,
        'Macro F1': f1_macro,
        'Lie F1': f1_lie,
        'Truth F1': f1_truth
    })

    # Printing the consolidated dataframe for quick analysis
    print("Model Comparison:")
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(args.save_path, 'model_comparison.csv'), index=False)

    # Plotting the comparison using a bar graph for each of the four models for more visualization
    if not args.no_plots:
        plt.figure(figsize=(14, 6))
        x = np.arange(len(models))
        width = 0.25

        plt.bar(x - width, f1_macro, width, label='Macro F1')
        plt.bar(x, f1_lie, width, label='Lie F1')
        plt.bar(x + width, f1_truth, width, label='Truth F1')

        plt.xlabel('Models')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_path, 'f1_comparison.png'))
        plt.show()

    # Using some hypothetical new messages to test the model
    messages_to_evaluate = [
        "I promise I won't attack your territory next turn.",
        "Let's form an alliance against Germany.",
        "I'm moving my troops to defend my own border, not to attack you.",
        "I need your help to defeat Russia, and then we'll share the spoils."
    ]

    # Using the best model (based on the comparison done above)
    best_model = sender_model_power
    best_vectorizer = sender_vectorizer_power

    print("\nEvaluating sample messages with the best model:")
    # Making the predictions using the best trained model for each of the new messages
    for i, message in enumerate(messages_to_evaluate):
        result = predict_lie(message, best_model, best_vectorizer, task="SENDER", 
                            use_power=True, power_threshold=args.power_threshold,
                            tokenizer_model=tokenizer_model)
        print(f"\nMessage {i+1}: \n{message}")
        print(f"Model's Prediction: {result['prediction']} (Model's Confidence in Prediction: {result['confidence']:.2f})")
        print(f"Probability that the message is a Lie: {result['probabilities']['lie']:.2f} || Probability that the message is a Truth: {result['probabilities']['truth']:.2f}")


if __name__ == "__main__":
    main()