# python script to automatically run classification and output identities for densities of interest
import multiprocessing
import numpy as np
import torch
from chimerax.core.commands import CmdDesc, run
from chimerax.atomic import selected_residues
from chimerax.core.commands import BoolArg, StringArg
from .models.MIC import MIC
from chimerax.core.commands import CmdDesc, run, BoolArg, StringArg

from .data.luna_utils import process_entries, select_proximal_data
from re import findall

def ui_mic(session, model_id, ions=None, extended=False, symmetry_expand=True, n_proc=8, fps_only=False):

    # Symmetry Expansion & FingerPrint Type Selection
    session.logger.info(f"model id {model_id}, ions {ions}, extended {extended}, symexp {symmetry_expand}, fps_only {fps_only}")
    fp_type = 'prune-eifp'
    torch_avail = torch.cuda.is_available()
    device = 'cuda' if torch_avail and not args.cpu_only else 'cpu'
    session.logger.info(f"Running MIC (prevalent set) with settings:\nFP Type: {fp_type}\n")
    session.logger.info(f'Running on GPU: {device=="cuda"}')
    if symmetry_expand:
        run(session, f'crystalcontacts #{model_id} d 6')
    session.logger.info(f"{model_id}, {ions}, {fps_only}")

    # Default Ion Selection
    ion_solvent_str = 'ions | solvent'
    if ions is None:
        sele_str = 'ions | solvent' # run on all ions
    else:
        sele_str = ions
    run(session, f'split #{model_id} atoms {ion_solvent_str}') 
    run(session, f'select #{model_id} & ({sele_str})')
    sites = selected_residues(session)
    if not sites:
        session.logger.error(f"No ions selected in model {model_id}. Make sure the ions of interest exist in your PDB file and that the PDB file of interest is uploaded.")
        return

    # Multi-Ion Selection [run MIC only on ions/solvents in PDB of interest ] 
    ion_like_residues = [r for r in sites if not r.polymer_type] #keep only ligands
    found_resnames = set(r.name.upper() for r in ion_like_residues)
    if ions:
        # Special case: user selected 'ions' (means all ion-like residues)
        if ions.strip().lower() == 'ions':
            sites = ion_like_residues
        else:
            requested_ions = set(findall(r'@(\w+)', ions.upper()))
            session.logger.info(f'requested ions: {requested_ions}')
            wants_solvent = 'solvent' in ions.lower()
            missing = requested_ions - found_resnames
            if missing:
                session.logger.warning(f"The following ions were not found in the structure: {missing}. If you're interested in a broader analysis, try selecting all ions or 'ions & solvents'.")
            filtered = [r for r in ion_like_residues if r.name.upper() in requested_ions]
            if wants_solvent:
                solvent_residues = [r for r in ion_like_residues if r.name.upper() == 'HOH']
                filtered.extend(solvent_residues)
            sites = filtered
    else:
        sites = ion_like_residues

    # FP Generation
    all_site_data = select_proximal_data(session, sites, model_id)
    ids, fps = process_entries(session, all_site_data)

    # MIC Prediction
    model = MIC(fp_type, extended, device=device)
    results = model.predict(np.log1p(np.stack(fps)), ids, return_proba=True)
    session.logger.info(results.to_string())
    
    # Label
    for site_id, prediction, confidence in zip(ids, results['prediction'], results['confidence']):
        chain, name, number = site_id.split(':')
        run(session, f'label #{model_id}/{chain}:{number} text {prediction}:{confidence}')
    return results.to_string()
    
mic_desc = CmdDesc(
    required=[('model_id', StringArg)],
    optional=[('ions', StringArg)],
    keyword=[('extended', BoolArg), ('symmetry_expand', BoolArg), ('fps_only', BoolArg)],
    synopsis='Run MIC analysis on a model.'
)