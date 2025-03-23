# NLP_Project

The QANTA Diplomacy project involves developing a model that predicts whether messages exchanged between players in the game Diplomacy are deceptive or truthful. The model analyzes in-game conversations and associated metadata to make its predictions. Its performance is evaluated based on how accurately it identifies deceptive and truthful messages.

# EXECUTION INSTRUCTIONS

BASELINE MODELS ARE STORED TILL NOW IN THE PYTHON SCRIPTS FOLDER :

## 1) BASELINE_BOW.PY

STEP - 1 : python baseline_BOW.py --data_path C:\Git\NLP_Project\NLP_Project\data --save_path models_BOW/ --max_iter 15 --power_threshold 4 (Training Command)

STEP - 2 : python predictions_BOW.py --model_path models_BOW/SENDER_with_power_model.pkl --vectorizer_path models_BOW/SENDER_with_power_vectorizer.pkl --message "I promise I won't attack your territory next turn." (For Inference)
