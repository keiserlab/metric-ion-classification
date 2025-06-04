#################################################################################
# Code for modified ion fingerprint generation using LUNA proximal interactions #
#################################################################################

import os
from ..lunasmall.interaction.utils import select_proximal
from ..lunasmall.interaction.filter import InteractionFilter, InteractionType
from ..lunasmall.interaction.filter import InteractionCalculator
from ..lunasmall.mol.groups import AtomGroupPerceiver, euclidean_distance
from chimerax.core.commands import run
from ..lunasmall.interaction.shells import ShellGenerator
from operator import le, ge
from ..lunasmall.mol.groups import AtomGroupPerceiver
from chimerax.core.commands import run
from ..lunasmall.interaction.utils import select_proximal

water = set(['HOH', 'DOD', 'WAT', 'H2O', 'OH', 'OH2', "O"])

def select_proximal_data(session, sites, model_id):
    '''Call select_proximal for each ion to fix their proximal ion/solvent and proximal residue selections before adding Hs for feature extraction
    '''
    all_site_data = []
    for site in sites:
        chain, resi = site.chain_id, site.number
        # select proximal
        ion = select_proximal(session, f'#{model_id}/{chain}:{resi}', 0.1, include_center=True)
        prox_ionsolvent, prox_res = select_proximal(session, f'#{model_id}/{chain}:{resi}', 6, split_ionwaters=True)
        all_site_data.append((site, ion, prox_ionsolvent, prox_res))
    return all_site_data

def process_entries(session, all_site_data):
    # add hydrogens once for the whole session
    run(session, 'addh hbond false')
    # Perform feature extraction using previously selected regions
    ids, fps, ion_names, prox_ionsolvent_names, prox_res_names = [], [], [], [], []
    for site, ion, prox_ionsolvent, prox_res in all_site_data:
        session.logger.info(f"Running feature extraction for {site.chain_id}:{site.name}:{site.number}")
        ion_info = (site.name, site.chain_id, site.number)
        prox_ionsolvent_info = [(res.name, res.chain_id, res.number) for res in prox_ionsolvent]
        prox_res_info = [(res.name, res.chain_id, res.number) for res in prox_res]
        ion_names.append(ion_info)
        prox_ionsolvent_names.append(prox_ionsolvent_info)
        prox_res_names.append(prox_res_info) 
        # Extract features
        trgt_grp, atm_grps_mngr = perceive_chemical_groups(session, prox_res, prox_ionsolvent, ion=ion, blind=True)
        if trgt_grp is None:
            continue
        inter_filter = InteractionFilter.new_pli_filter(ignore_self_inter=True,
                                                        ignore_intra_chain=False,
                                                        ignore_any_h2o=False, 
                                                        ignore_h2o_h2o=False)
        proximal_only = {("Atom", "Atom"): [calc_proximal]}
        inter_calc = InteractionCalculator(inter_filter=inter_filter, 
                                           inter_funcs=proximal_only,
                                           add_h2o_pairs_with_no_target=True)  
        interactions_mngr = inter_calc.calc_interactions([trgt_grp], nb_atm_grps=atm_grps_mngr.atm_grps)    
        atm_grps_mngr.add_atm_grps([trgt_grp]) 
        fp = create_ifp(atm_grps_mngr)
        ids.append(f"{site.chain_id}:{site.name}:{site.number}")
        fps.append(fp)
    # Clean-up
    run(session, "delete H")  
    return ids, fps
 
def perceive_chemical_groups(session, prox_res, prox_ions, ion = None, blind=True):    

    file_dir = os.path.dirname(os.path.realpath(__file__))
    perceiver = AtomGroupPerceiver(feature_extractor=None, tmp_path = '/tmp')#_get_perceiver(options)n
    check = perceiver.perceive_atom_groups(prox_res, session=session)

    if check is None:
        return None, None

    perceiver.add_dummy_groups(prox_ions, target = False)
    trgt_grp, atm_grps_mngr = perceiver.add_dummy_groups(ion, target = True)
    return trgt_grp, atm_grps_mngr

def create_ifp(atm_grps_mngr):

    sg = ShellGenerator(18, 0.25,
                        diff_comp_classes=False,
                        ifp_type='eifp')
    sm = sg.create_shells(atm_grps_mngr)
    return sm.to_fingerprint(4096, count_fp=True)

def calc_proximal(self, params):
    group1, group2, feat1, feat2 = params
    interactions = []
    cc_dist = euclidean_distance(group1.centroid, group2.centroid)
    if (self.is_within_boundary(cc_dist, "min_dist_proximal", ge)
            and self.is_within_boundary(cc_dist, "max_dist_proximal", le)):

        params = {"dist_proximal": cc_dist}
        inter = InteractionType(group1, group2, "Proximal", params=params)
        interactions.append(inter)
    return interactions

def get_restype(res, session):
    session.logger.info(str(res))
    session.logger.info(res.name)
    session.logger.info(str(res.description))

def sym_expand(pdb_file, cutoff=7):

    outfile = pdb_file
    
    if len(os.path.dirname(pdb_file)) > 0:
        outfile = os.path.dirname(pdb_file) + '/' + os.path.basename(pdb_file).lower()
    return outfile.lower()
