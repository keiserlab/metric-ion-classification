# python script to automatically run classification and output identities for densities of interest
from chimerax.core.commands import CmdDesc, run
from chimerax.atomic import selected_residues
from chimerax.core.commands import BoolArg, StringArg, FileNameArg
import re
import numpy as np
import torch
from re import findall
from .models.MIC import MIC
from .data.luna_utils import process_entries, select_proximal_data

def mic(session, selection_str, extended=False, symmetry_expand=True, fps_only=False, save_as=None):
    
    # Model Selection & Symmetry Expansion & FingerPrint Type Selection
    session.logger.info(f"Selection input: {selection_str}")
    session.logger.info(f"Extended: {extended}, Symmetry Expand: {symmetry_expand}, FPS Only: {fps_only}")
    model_match = re.match(r"#(\d+)", selection_str.strip())
    if not model_match:
        session.logger.error("Failed to parse model ID from selection string.")
        return
    model_id = model_match.group(1)
    session.logger.info(f"Model ID: {model_id}")
    fp_type = 'prune-eifp'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    session.logger.info(f"FP Type: {fp_type}, Device: {device}")
    if symmetry_expand:
        run(session, f'crystalcontacts #{model_id} d 6')
        session.logger.info(f"Symmetry expanded #{model_id}")

    # Ion Selection
    run(session, f'split #{model_id} atoms ions | solvent')
    session.logger.info(f"Split ions/solvent for #{model_id}")
    default_selection = selection_str.strip() == f"#{model_id}"
    if default_selection: #default selection
        sele_str = "ions | solvent"
        run(session, f'select #{model_id} & ({sele_str})')
        session.logger.info("Default selection: all ions and solvent")
    else: #user's custom selection
        run(session, f'select {selection_str}')
        session.logger.info(f"Custom selection: {selection_str}")
    sites = selected_residues(session)
    session.logger.info(f"Residues selected: {len(sites)}")
    if not sites:
        session.logger.error("No residues selected. Check your command or PDB file.")
        return

    ion_like_residues = [r for r in sites if not r.polymer_type] #keep only ligands
    found_resnames = set(r.name.upper() for r in ion_like_residues)
    if not ion_like_residues:
        session.logger.warning("No valid ion or solvent residues in selection.")
        return

    if default_selection: #mic model_id
        filtered = [r for r in ion_like_residues]
        session.logger.info(f"Default behavior: using all {len(filtered)} ion-like residues in model.")
    else:
        raw = selection_str.split(':')[-1] if ':' in selection_str else ''
        requested = [x.strip().upper() for x in raw.split(',')] if raw else []
        requested_ions = {r for r in requested if r != 'SOLVENT'}
        wants_solvent = 'SOLVENT' in requested
        # No specific ions? default to all ions in specified chain
        if not requested_ions:
            filtered = [r for r in ion_like_residues]
        else:
            filtered = [r for r in ion_like_residues if r.name.upper() in requested_ions]
            found_resnames = set(r.name.upper() for r in ion_like_residues)
            missing = requested_ions - found_resnames
            if missing:
                session.logger.warning(f"The following ions were not found: {missing}")
        # Add HOH from model if solvent requested
        if wants_solvent:
            run(session, f'select add #{model_id} & solvent')
            extra_solvent = selected_residues(session)
            hoh_residues = [r for r in extra_solvent if r.name.upper() == 'HOH']
            session.logger.info(f"Found {len(hoh_residues)} HOH residues in model #{model_id}")
            filtered.extend(hoh_residues)
    sites = filtered

    session.logger.info(f"Final site count: {len(sites)}")
    if not sites:
        session.logger.error("No valid residues left after filtering.")
        return
    for r in sites:
        session.logger.info(f"Selected site -> Chain: {r.chain_id}, ResName: {r.name}, ResNum: {r.number}")

    # Fingerprint generation
    all_site_data = select_proximal_data(session, sites, model_id)
    ids, fps = process_entries(session, all_site_data)
    session.logger.info(f"Processed {len(fps)} fingerprints")

    mic_model = MIC(fp_type, extended, device=device)
    results = mic_model.predict(np.log1p(np.stack(fps)), ids, return_proba=True)
    session.logger.info("MIC prediction complete")
    session.logger.info(results.to_string())

    # Label results
    for site_id, prediction, confidence in zip(ids, results['prediction'], results['confidence']):
        chain, name, number = site_id.split(':')
        run(session, f'label #{model_id}/{chain}:{number} text {prediction}:{confidence}')
        session.logger.info(f"Labeled {site_id} with {prediction}:{confidence}")

    if save_as:
        try:
            results.to_csv(save_as) # , index=False
            session.logger.info(f"MIC predictions saved to: {save_as}")
        except Exception as e:
            session.logger.error(f"Failed to save MIC results to {save_as}: {e}")

    return results.to_string()


mic_desc = CmdDesc(
    required=[('selection_str', StringArg)],
    keyword=[
        ('extended', BoolArg),
        ('symmetry_expand', BoolArg),
        ('fps_only', BoolArg),
        ('save_as', FileNameArg)  
    ],
    synopsis='Run MIC on model or selection, optionally save predictions to CSV.'
)

