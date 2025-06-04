
from collections import defaultdict

class ChemicalFeature():
    def __init__(self, name):
        self.name = name

    def format_name(self, case_func="sentencecase"):
        """Convert chemical feature names to another string case.

        Parameters
        ----------
        name : str
            The name of a string case function from
            :py:mod:`luna.util.stringcase`.
        """

        func = getattr(case, case_func)
        return func(self.name)

    # Special methods
    def __repr__(self):
        return "<Feature=%s>" % self.name

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(self, other.__class__):
            return self.name == other.name
        return False

    def __ne__(self, other):
        """Overrides the default implementation"""
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)


class FeatureExtractor:
    def __init__(self, feature_factory):
        self.feature_factory = feature_factory

    def get_features_by_atoms(self, mol_obj, atm_map=None):

        perceived_features = \
                self.feature_factory.GetFeaturesForMol(mol_obj)

        atm_features = defaultdict(set)
        for f in perceived_features:
            for atm_idx in f.GetAtomIds():
                tmp_atm_idx = atm_idx
                if atm_map is not None:
                    if atm_idx in atm_map:
                        tmp_atm_idx = atm_map[atm_idx]
                    else:
                        logger.warning("There is no corresponding mapping to "
                                       "the index '%d'. It will be ignored."
                                       % atm_idx)

                feature = ChemicalFeature(f.GetFamily())
                atm_features[tmp_atm_idx].add(feature)

        return atm_features

    def chimera_to_smiles(self, atms):
        """
        Converts collection of ChimeraX atom objects to corresponding SMILES string
        """
        

    def get_features_for_chimerax(self, atms):
        grp_features = defaultdict(set)
        for key, smarts in self.feature_factory.GetFeatureDefs().items():
            grp_type = key.split(".")[0]

            ob_smart.Match(ob_mol)




    def get_features_by_groups(self, mol_obj, atm_map=None):
        perceived_features = \
                self.feature_factory.GetFeaturesForMol(mol_obj)

        grp_features = {}
        for f in perceived_features:
            atm_ids = sorted(list(f.GetAtomIds()))

            if atm_map is not None:
                tmp_atm_ids = []
                for atm_id in atm_ids:
                    if atm_id in atm_map:
                        tmp_atm_ids.append(atm_map[atm_id])
                    else:
                        logger.warning("There is no corresponding mapping to "
                                       "the index '%d'. It will be ignored."
                                       % atm_id)
                atm_ids = tmp_atm_ids

            key = ','.join([str(x) for x in atm_ids])
            if key in grp_features:
                grp_obj = grp_features[key]
            else:
                grp_obj = {"atm_ids": atm_ids, "features": []}

            features = set(grp_obj["features"])
            feature = ChemicalFeature(f.GetFamily())
            features.add(feature)
            grp_obj["features"] = list(features)
            grp_features[key] = grp_obj

        return grp_features