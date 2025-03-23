# NLP_Project

The QANTA Diplomacy project involves developing a model that predicts whether messages exchanged between players in the game Diplomacy are deceptive or truthful. The model analyzes in-game conversations and associated metadata to make its predictions. Its performance is evaluated based on how accurately it identifies deceptive and truthful messages.

# EXECUTION INSTRUCTIONS

BASELINE MODELS ARE STORED TILL NOW IN THE PYTHON SCRIPTS FOLDER :

## 1) BAG OF WORDS PYTHON SCRIPTS

STEP - 1 : cd python_scripts

STEP - 2 : python baseline_BOW.py --data_path C:\Git\NLP_Project\NLP_Project\data --save_path models_BOW/ --max_iter 15 --power_threshold 4 (Training Command)

STEP - 3 : python predictions_BOW.py --model_path models_BOW/SENDER_with_power_model.pkl --vectorizer_path models_BOW/SENDER_with_power_vectorizer.pkl --message "I promise I won't attack your territory next turn." (For Inferenence)

## 2) CONTEXT-LSTM + POWER MODEL

STEP - 1 : cd python_scripts

STEP - 2 : python baseline_context_LSTM.py

STEP - 3 : python predictions_context_LSTM.py --model_path models_lstm/best_model.pt --sample_message "I promise I won't attack your territory next turn." --power_delta 4

## 3) Models Drive Link

Drive Link - https://drive.google.com/drive/folders/1zNyJ8Cs1Vzt1ohzljB5PHJHfILcK5Xcz?usp=sharing
