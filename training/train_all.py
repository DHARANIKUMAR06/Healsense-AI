import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from datasets.generate_datasets import main as generate_main
from training.train_diabetes import train_diabetes
from training.train_heart import train_heart
from training.train_kidney import train_kidney
from training.train_malaria import train_malaria
from training.train_dengue import train_dengue

def train_all_models():
    print("===========================================")
    print("HealthSense AI - Training All 5 ML Models")
    print("===========================================")
    
    train_diabetes()
    train_heart()
    train_kidney()
    train_malaria()
    train_dengue()
    
    print("\nAll 5 Disease Models successfully trained and saved into models/")

if __name__ == '__main__':
    train_all_models()
