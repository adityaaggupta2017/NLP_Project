#!/usr/bin/env python

import subprocess
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Setup for Deception Detection with ConceptNet")
    parser.add_argument('--download_conceptnet', action='store_true', help='Download ConceptNet Numberbatch embeddings')
    parser.add_argument('--download_spacy', action='store_true', help='Download spaCy model')
    parser.add_argument('--setup_all', action='store_true', help='Setup everything')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Install requirements
    print("Installing Python dependencies...")
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    
    # Download spaCy model
    if args.download_spacy or args.setup_all:
        print("Downloading spaCy model...")
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    
    # Download ConceptNet Numberbatch embeddings
    if args.download_conceptnet or args.setup_all:
        print("Downloading ConceptNet Numberbatch embeddings...")
        conceptnet_url = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
        output_gz = "data/numberbatch-en.txt.gz"
        output_file = "data/numberbatch-en.txt"
        
        # Download file
        subprocess.run(['curl', '-L', conceptnet_url, '-o', output_gz])
        
        # Extract file
        print("Extracting ConceptNet embeddings...")
        subprocess.run(['gunzip', '-c', output_gz, '>', output_file], shell=True)
        
        # Remove gz file to save space
        os.remove(output_gz)
    
    print("Setup completed successfully!")

if __name__ == "__main__":
    main()