#################################################################################
# Code for modified ion fingerprint generation using LUNA proximal interactions #
#################################################################################

import os
import os.path as osp

from rdkit.Chem import ChemicalFeatures

import luna
from luna.MyBio.PDB.PDBParser import PDBParser
from luna.MyBio.util import get_entity_from_entry
from luna.mol.features import FeatureExtractor, ChemicalFeature
from luna.interaction.contact import get_contacts_with
from luna.interaction.filter import InteractionFilter
from luna.interaction.calc import InteractionCalculator, InteractionsManager
from luna.interaction.type import InteractionType
from luna.interaction.fp.type import IFPType
from luna.interaction.fp.shell import ShellGenerator
from luna.util.default_values import *
import luna.util.math as im
from operator import le, ge

import logging
mic_logger = logging.getLogger('mic_logger')

from .luna_overrides import *

from pymol import cmd
cmd.feedback("disable","all","actions")
cmd.feedback("disable","all","results")

def process_entry(entry, prune = True, ifp_type = 'eifp', 
                  ifp_num_levels = 18,
                  ifp_radius_step = 0.25,
                  ifp_length = 4096,
                  ifp_count = True,
                  pdb_path = "./",
                  blind=True,
                  normfunc = None,
                  flag_distance = 4,
                  flag_threshold = 3,  
                  return_bitinfo = False,
                  return_atm_mngr = False):
    """
    Calculate density-IFP 
    """
    
    try:
        # setup options so LUNA is happy
        opt = {}
        opt['add_h'] = True
        opt['amend_mol'] = False
        opt['pdb_path'] = pdb_path
        opt['ph'] = 7.4
        opt["verbosity"] = 0
        #ifp argss
        if ifp_type == 'eifp':
            opt['ifp_type'] = IFPType.EIFP
        else:
            opt['ifp_type'] = IFPType.FIFP

        opt['ifp_count'] = ifp_count
        opt['ifp_diff_comp_classes'] = False

        opt['ifp_num_levels'] = ifp_num_levels
        opt['ifp_radius_step'] = ifp_radius_step
        opt["ifp_length"] = ifp_length

        inter_filter = InteractionFilter.new_pli_filter(ignore_self_inter=True,
                                                        ignore_intra_chain=False,
                                                        ignore_any_h2o=False, 
                                                        ignore_h2o_h2o=False)
        proximal_only = {("Atom", "Atom"): [calc_proximal]}
        inter_calc = InteractionCalculator(inter_filter=inter_filter, 
                                           add_proximal = True, 
                                           inter_funcs = proximal_only,
                                           add_h2o_pairs_with_no_target = True) # important for water site
        opt['inter_calc'] = inter_calc
        opt["atom_prop_file"] = ATOM_PROP_FILE

        pdb_file = os.path.join(pdb_path, entry.pdb_id + '.pdb')

        pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True, FIX_EMPTY_CHAINS=True,
                                           FIX_ATOM_NAME_CONFLICT=True, FIX_OBABEL_FLAGS=False)
        structures = load_all_structures(pdb_file, pdb_parser)
        #structure = pdb_parser.get_structure(entry.pdb_id, os.path.join(pdb_path, entry.pdb_id + '.pdb'))
        opt['add_h']  = _decide_hydrogen_addition(opt['add_h'], pdb_parser.get_header(), entry)

        ligand = get_entity_from_entry(structures[0][0], entry)
        ligand.set_as_target(is_target=True)

        atm_grps_mngr = _perceive_chemical_groups(structures, ligand, opt, blind=blind)
        cluster_members = count_neighbors(structures, ligand, thresh_radius=flag_distance)

        atm_grps_mngr.entry = entry

        # find proximal interactions
        interactions_mngr = opt["inter_calc"].calc_interactions(atm_grps_mngr.atm_grps)
        interactions_mngr.entry = entry            
        
        atm_grps_mngr.merge_hydrophobic_atoms(interactions_mngr)

        # Prune to only our target
        if prune:
            all_ints_to_keep = []
            # force ONLY consideration of protein/ligand/non-water contacts + ligand
            # need to remove these from that atom_manager, since that's what's actually used to
            # calculate the fingerprints
            for atm_grp in atm_grps_mngr:
                ints_to_keep = []
                for inter in atm_grp.interactions:
                    if inter.src_grp.has_target() or inter.trgt_grp.has_target():
                        ints_to_keep.append(inter)
                        all_ints_to_keep.append(inter)
                atm_grp.interactions = ints_to_keep
            interactions_mngr = InteractionsManager(set(all_ints_to_keep))  

        # Generate IFP
        ifp = None
        ifp, bit_info = _create_ifp(opt, atm_grps_mngr, return_bitinfo=return_bitinfo)
        ifp = ifp.to_vector(compressed = False) 
    
        if normfunc is not None:
            ifp = normfunc(ifp)

        if not return_atm_mngr and not return_bitinfo:
            return (ifp, None, len(cluster_members) > flag_threshold)
        
        to_return = [(ifp, None, len(cluster_members) > flag_threshold)]

        if return_atm_mngr:
            to_return.append(atm_grps_mngr)

        if return_bitinfo:
            to_return.append(bit_info)

        return to_return

    except Exception as e:
        print(f"Exception triggered for {entry}", e)
        return None, e


def count_neighbors(structures, ligand, thresh_radius):
    nb_pairs = set()
    for struct in structures:
        new_pairs = get_contacts_with(struct[0], ligand, level='R', radius=thresh_radius)
        nb_pairs = nb_pairs | new_pairs

    nb_compounds = set([x[0] for x in nb_pairs])
    cluster_members = set()

    for x in nb_compounds:
        if x.is_water() or x.is_metal() or (x.is_hetatm() and len(x) == 1):
            cluster_members.add(x)
    return cluster_members

def _perceive_chemical_groups(structures, ligand, options, blind=True):
    inter_config = options["inter_calc"].inter_config
    radius = inter_config.get("bsite_cutoff", BOUNDARY_CONFIG["bsite_cutoff"])
    nb_pairs = set()
    for struct in structures:
        new_pairs = get_contacts_with(struct[0], ligand, level='R', radius=radius)
        nb_pairs = nb_pairs | new_pairs
    
    mol_objs_dict = {}

    #MIC update - remove all waters, ions, metals from pairs - DON'T use them to 
    # calculate chemical properties

    nb_compounds = set([x[0] for x in nb_pairs])


    perceiver = _get_perceiver(options)

    if blind:
        final_nb_compounds = set()
        removed_compounds = set()

        for x in nb_compounds:
            if (x.is_hetatm() and len(x) == 1) or x.is_metal() or x.is_water():
                removed_compounds.add(x)
            else:
                final_nb_compounds.add(x)
        perceiver.perceive_atom_groups(final_nb_compounds, mol_objs_dict=set([r.id for r in removed_compounds]))
        atm_grps_mngr = perceiver.add_dummy_groups(removed_compounds)

    else:
        atm_grps_mngr = perceiver.perceive_atom_groups(nb_compounds, {})

    return atm_grps_mngr

def _get_perceiver(options):
    feats_factory_func = ChemicalFeatures.BuildFeatureFactory
    feature_factory = feats_factory_func(options["atom_prop_file"])
    feature_extractor = FeatureExtractor(feature_factory)

    perceiver = MICPerceiver(feature_extractor, add_h=options['add_h'],
                                       ph=options["ph"], amend_mol=options["amend_mol"],
                                       tmp_path="/tmp")
    return perceiver

def _create_ifp(options, atm_grps_mngr, return_bitinfo = False):
    sg = ShellGenerator(options["ifp_num_levels"], options["ifp_radius_step"],
                        diff_comp_classes=options["ifp_diff_comp_classes"],
                        ifp_type=options["ifp_type"])
    sm = sg.create_shells(atm_grps_mngr)

    unique_shells = not options["ifp_count"]

    if return_bitinfo:
        bit_info = {}
        fp = sm.to_fingerprint(fold_to_length=options["ifp_length"],
                             unique_shells=unique_shells,
                             count_fp=options["ifp_count"])
        for fp_on in fp.indices:
            bit_info[fp_on] = list(sm.trace_back_feature(fp_on, fp, unique_shells=unique_shells))
        return fp, bit_info 

    return sm.to_fingerprint(fold_to_length=options["ifp_length"],
                             unique_shells=unique_shells,
                             count_fp=options["ifp_count"]), None

def _decide_hydrogen_addition(add_h, pdb_header, entry):
        if add_h:
            if "structure_method" in pdb_header:
                method = pdb_header["structure_method"]
                # If the method is not a NMR type does not add hydrogen
                # as it usually already has hydrogens.
                if method.upper() in NMR_METHODS:
                    return False
            return True
        return False

def calc_proximal(self, params):
    group1, group2, feat1, feat2 = params
    interactions = []
    cc_dist = im.euclidean_distance(group1.centroid, group2.centroid)
    if (self.is_within_boundary(cc_dist, "min_dist_proximal", ge)
            and self.is_within_boundary(cc_dist, "max_dist_proximal", le)):

        params = {"dist_proximal": cc_dist}
        inter = InteractionType(group1, group2, "Proximal", params=params)
        interactions.append(inter)

    return interactions


def sym_expand(pdb_file, cutoff=7):
  
    cmd.reinitialize()

    pdbid = os.path.basename(pdb_file)
    pdbdir = os.path.dirname(pdb_file)
    
    try:
        cmd.load(pdb_file, 'prot') # load pdb
    except:
        mic_logger.info(f"PDB file {pdb_file}, not found, fetching...")
        cmd.set('fetch_path', pdbdir)
        cmd.fetch(pdbid.replace('.pdb','').lower(), type='pdb', name='prot')
        pdbid = pdbid.lower()

    if cutoff < 0:
        return osp.join(pdbdir, pdbid) 
 
    pdbid = pdbid.replace('.pdb', '.symexp.pdb')

    cmd.select('ions', '(inorganic or resn HOH)') # select all ions and water
    cmd.symexp('sym', 'prot', 'prot', 6)
    cmd.select('selection', f'prot or all within {cutoff} of ions')
    cmd.multisave(osp.join(pdbdir, pdbid), 'selection')
  
    mic_logger.debug(f"Symmetry expanded PDB file saved to {osp.join(pdbdir, pdbid)}.") 
    return osp.join(pdbdir, pdbid)
