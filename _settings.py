import getpass
import os
import sys

__USERNAME = getpass.getuser()

# Set the following:
DATA_PATH = "/srv/local/data"
MIMIC_PATH = os.path.join(DATA_PATH, 'MIMIC-III', 'mimic-iii-clinical-database-1.4') # put MIMIC data to here
NEPTUNE_PROJECT = ''
NEPTUNE_API_TOKEN = ""


# No need to change the following
WORKSPACE = os.path.join(f'{DATA_PATH}/{__USERNAME}/FavMac/', "Temp")
os.makedirs(WORKSPACE, exist_ok=True)

MIMIC_PREPROCESS_OUTPUT = os.path.join(WORKSPACE, 'mimic-iii-processed')

MIMICIIICompletion_NAME = 'MIMIC-III' + 'Completion'
CLAIMDEMO_NAME = 'ClaimPredNew'
CLAIMDEMOSeq_NAME = CLAIMDEMO_NAME + 'Seq'
MNISTSup_NAME = "MNISTSup"

CLAIMDEMO_PATH = ""
MNIST_PATH = os.path.join(DATA_PATH, 'MNIST')

LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
RANDOM_SEED = 7

NCOLS = 80

assert NEPTUNE_PROJECT != '', "Please set NEPTUNE_PROJECT"
assert NEPTUNE_API_TOKEN != '', "Please set NEPTUNE_API_TOKEN"