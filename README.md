# NLP_Project

The QANTA Diplomacy project aims to develop a model that predicts whether messages exchanged between players in the game *Diplomacy* are deceptive or truthful. The project uses in-game conversations and associated metadata to make predictions, with evaluation based on how accurately deceptive and truthful messages are identified.

---

## Table of Contents

- [Overview](#overview)
- [Execution Instructions](#execution-instructions)
  - [Baseline Models](#baseline-models)
    - [Bag of Words Model](#bag-of-words-model)
    - [Context-LSTM + Power Model](#context-lstm--power-model)
  - [Novel Models](#novel-models)
    - [Novel Model without ConceptNet](#novel-model-without-conceptnet)
    - [Novel Model with ConceptNet](#novel-model-with-conceptnet)
- [Drive Link](#drive-link)

---

## Overview

This repository contains the code for several models developed as part of the QANTA Diplomacy project. Both baseline and novel models are provided, with support for inference through command line scripts and Jupyter Notebook (.ipynb) files.

---

## Execution Instructions

### Baseline Models

The baseline models are stored in the `python_scripts` folder. They include a Bag of Words model and a Context-LSTM model.

#### Bag of Words Model

1. **Navigate to the `python_scripts` folder:**

   ```bash
   cd python_scripts
2. **Train the model:** 

  ```bash 
  python baseline_BOW.py --data_path C:\Git\NLP_Project\NLP_Project\data --save_path models_BOW/ --max_iter 15 --power_threshold 4

3. Run inference: 

  ```bash 
  python predictions_BOW.py --model_path models_BOW/SENDER_with_power_model.pkl --vectorizer_path models_BOW/SENDER_with_power_vectorizer.pkl --message "I promise I won't attack your territory next turn."
 
#### Context-LSTM + Power Model 

1. **Navigate to the `python_scripts` folder:** 
  ```bash
  cd python_scripts
2. **Train the model:** 

  ```bash 
  python baseline_context_LSTM.py

3. **Run inference:** 

  ```bash 
  python predictions_context_LSTM.py --model_path models_lstm/best_model.pt --sample_message "I promise I won't attack your territory next turn." --power_delta 4

### Novel Models 

The novel models are stored in the folder novel_python_scripts and come in two versions: with and without ConceptNet. 

#### Novel Model without ConceptNet 

1. **Navigate to the `novel_python_scripts` folder:** 

  ```bash  
   cd novel_python_scripts/Without_ConceptNet

2. **Train the model:** 

  ```bash 
   python train.py --train_path C:\Git\NLP_Project\NLP_Project\data\train.jsonl --val_path C:\Git\NLP_Project\NLP_Project\data\validation.jsonl --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --model_name roberta-base --batch_size 32 --epochs 5 --lr 5e-6 --use_game_scores --oversample_factor 30 --truth_focal_weight 4.0 --gradient_accumulation_steps 2 --output_dir outputs
 
3. **Run inference:** 

  ```bash 
   python inference.py --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --model_path C:\Git\NLP_Project\NLP_Project\novel_models\kaggle\working\best_macro_f1_model.pt --model_name roberta-base --batch_size 32 --use_game_scores --output_file predictions.jsonl


#### Novel Model with ConceptNet 

1. **Navigate to the `novel_python_scripts` folder:** 

  ```bash 
   cd novel_python_scripts/With_ConceptNet
2. **Set up the required dependencies:**
  ```bash 
   python setup.py --setup_all

3. **Train the model:** 
  ```bash 
   python train.py --train_path C:\Git\NLP_Project\NLP_Project\data\train.jsonl --val_path C:\Git\NLP_Project\NLP_Project\data\validation.jsonl --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --conceptnet_path data/numberbatch-en.txt --model_name roberta-base --batch_size 32 --epochs 5 --lr 5e-6 --use_game_scores --oversample_factor 30 --truth_focal_weight 4.0 --gradient_accumulation_steps 2 --output_dir outputs

4. **Run inference:** 
  ```bash 
   python inference.py --test_path C:\Git\NLP_Project\NLP_Project\data\test.jsonl --model_path "C:\Git\NLP_Project\NLP_Project\novel_models(With Conceptnet)\kaggle\working\best_macro_f1_model.pt" --conceptnet_path data/numberbatch-en.txt --model_name roberta-base --batch_size 32 --use_game_scores --output_file predictions.jsonl


### Drive Link 
You can access the models and additional data via the following Google Drive link: 

https://drive.google.com/drive/folders/1zNyJ8Cs1Vzt1ohzljB5PHJHfILcK5Xcz?usp=sharing

