import pytest

import os
from glob import glob

import numpy as np
from io import StringIO

from mic.fingerprints.FPParser import FPParser
from mic.models.MIC import MIC

def test_xray_file():
    os.mkdir('tests/tmp')

    fpparser = FPParser('tests/tmp/4OKE.pdb', 'prune-eifp', n_proc=10)
    mic_model = MIC('prune-eifp', False, device='cpu')
    preds = mic_model.predict(fpparser.fps, fpparser.entries, return_proba = False)

    max_mg_entry = preds.loc[preds['prediction'] == 'MG'].sort_values('confidence', ascending=False).index[0]
    assert max_mg_entry._chain_id == 'A'
    assert max_mg_entry._comp_name == 'MG'
    assert max_mg_entry._comp_num == 202

def test_cryoem_file():
    
    fpparser = FPParser('7A4M.pdb', 'prune-eifp', n_proc=10, symexp_cutoff = -1)
    mic_model = MIC('prune-eifp', False, device='cpu')
    preds = mic_model.predict(fpparser.fps, fpparser.entries, return_proba = False)

    max_zn_entry = preds.loc[preds['prediction'] == 'ZN'].sort_values('confidence', ascending=False).index[0]
    assert max_zn_entry._chain_id == 'A'
    assert max_zn_entry._comp_name == 'ZN'
    assert max_zn_entry._comp_num == 202

    os.remove('7a4m.pdb')

def test_xray_entries():
    xray_entry_str = '4OKE,A,MG,202,\n3A09,A,CA,601,\n4L9P,B,ZN,601,'
    xray_entry_strio = StringIO(xray_entry_str)

    fpparser = FPParser('./', 'prune-eifp', entry_file=xray_entry_strio, n_proc = 1)

    mic_model = MIC('prune-eifp', False, device='cpu')
    preds = mic_model.predict(fpparser.fps, fpparser.entries, return_proba = False)

    assert np.array_equal(preds['prediction'].values, ['MG','CA','ZN'])
    assert np.array_equal(preds['confidence'].values, [0.97,0.8987,1.]) 

    for f in glob('*.pdb'):
        os.remove(f)

def test_cryoem_entries():
    mc4r_entry_str = "7aue,R,CA,501,X\n7f53,R,CA,601,X\n7f55,R,CA,601,X\n7f54,R,CA,601,X\n7f58,R,CA,602,X\n7piu,R,CA,401,X\n7piv,R,CA,401,X"
    mc4r_entry_strio = StringIO(mc4r_entry_str)

    fpparser = FPParser('./tests/tmp', 'prune-eifp', entry_file=mc4r_entry_strio, n_proc = 10)

    assert fpparser.fps.shape == (7,4096)

    mic_model = MIC('prune-eifp', False, device='cpu')
    preds = mic_model.predict(fpparser.fps, fpparser.entries, return_proba = False)

    assert np.array_equal(preds['prediction'].values, ['CA','CA','CA','CA','CA','HOH','CA'])
    assert np.array_equal(preds['confidence'].values, [0.665,0.9468,0.8896,0.8313,0.8614,0.9051,0.3574])

    for f in glob('tests/tmp/*.pdb'):
        os.remove(f)
    os.rmdir('tests/tmp')

