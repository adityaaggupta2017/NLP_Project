# NLP_Project

The QANTA Diplomacy project involves developing a model that predicts whether messages exchanged between players in the game Diplomacy are deceptive or truthful. The model analyzes in-game conversations and associated metadata to make its predictions. Its performance is evaluated based on how accurately it identifies deceptive and truthful messages.

# EXECUTION INSTRUCTIONS
The ipynb files for the 2 baselines and the novel models are there also . They can be directly run for inference . The python scripts for the same are as follows : 

1) BASELINE MODELS :  
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

2) NOVEL MODELS : 

  THE NOVEL PYTHON SCRIPTS ARE STORED IN THE FOLDER NAMED novel_python_scripts . The running commands for both the models(with and without conceptnet) are as follows: 

  ## 1) NOVEL MODEL WITHOUT CONCEPTNET 

  STEP - 1 : cd novel_python_scripts/Without_ConceptNet 

  STEP - 2 : python train.py --train_path C:\Git\NLP_Project\NLP_Project\data\train.jsonl --val_path C:\Git\NLP_Project\NLP_Project\data\validation.jsonl --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --model_name roberta-base --batch_size 32 --epochs 5 --lr 5e-6 --use_game_scores --oversample_factor 30 --truth_focal_weight 4.0 --gradient_accumulation_steps 2 --output_dir outputs     (FOR TRAINING) 

  STEP - 3 : python inference.py --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --model_path C:\Git\NLP_Project\NLP_Project\novel_models\kaggle\working\best_macro_f1_model.pt --model_name roberta-base --batch_size 32 --use_game_scores --output_file predictions.jsonl    (FOR INFERENCE) 


  ## 2) NOVEL MODEL WITH CONCEPTNET 

  STEP - 1 : cd novel_python_scripts/With_ConceptNet 

  STEP - 2 : python setup.py --setup_all 

  STEP - 3 : python train.py --train_path C:\Git\NLP_Project\NLP_Project\data\train.jsonl --val_path C:\Git\NLP_Project\NLP_Project\data\validation.jsonl --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --conceptnet_path data/numberbatch-en.txt --model_name roberta-base --batch_size 32 --epochs 5 --lr 5e-6 --use_game_scores --oversample_factor 30 --truth_focal_weight 4.0 --gradient_accumulation_steps 2 --output_dir outputs   (FOR TRAINING)

  STEP - 4 : python inference.py --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --model_path C:\Git\NLP_Project\NLP_Project\novel_models(With Conceptnet)\kaggle\working\best_macro_f1_model.pt --conceptnet_path data/numberbatch-en.txt --model_name roberta-base --batch_size 32 --use_game_scores --output_file predictions.jsonl    (FOR INFERENCE) 

Drive Link - https://drive.google.com/drive/folders/1zNyJ8Cs1Vzt1ohzljB5PHJHfILcK5Xcz?usp=sharing
