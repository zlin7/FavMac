import os

import ipdb
import persist_to_disk as ptd

import _settings
from _settings import MIMIC_PATH, MIMIC_PREPROCESS_OUTPUT


def _extract_subjects():
    import subprocess
    args = ['python', '-m', 'data_utils.preprocessing.mimic3benchmark.extract_subjects']
    args.extend([MIMIC_PATH, MIMIC_PREPROCESS_OUTPUT])
    cwd = os.path.dirname(os.path.abspath(_settings.__file__))
    print(args, cwd)
    if os.path.isdir(MIMIC_PREPROCESS_OUTPUT) and len(os.listdir(MIMIC_PREPROCESS_OUTPUT)) == 33801:
        print("already extracted")
        return
    if os.path.isdir(MIMIC_PREPROCESS_OUTPUT):
        assert len(os.listdir(MIMIC_PREPROCESS_OUTPUT)) <= 33801, "?"
    subprocess.call(args, cwd=cwd)

@ptd.persistf()
def _extract_times_series():
	args = ['python', '-m', 'data_utils.preprocessing.mimic3benchmark.extract_episodes_from_subjects']
	args.extend([MIMIC_PREPROCESS_OUTPUT])

def _cache_code():
    import data_utils.mimic3.code as code
    code.MIMICIII_DiagCodes_Reader()
    code.MIMICIII_DiagCodes_Reader(keep_proba=0.)

def _cache_events():
    import data_utils.mimic3.events as events
    events.MIMIC_III_Event_Reader()

def _cache_notes():
    import data_utils.preprocessing.clinicalBERT_preprocess as clinicalBERT_preprocess
    clinicalBERT_preprocess.main()
    import data_utils.mimic3.notes as notes
    notes.MIMIC_III_Notes_BertFeature_Reader()

if __name__ == '__main__':
    _extract_subjects()
    _cache_code()
    _cache_events()
    _cache_notes()