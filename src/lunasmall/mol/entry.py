class Entry:
    def __init__(self, pdb_id, chain_id, comp_name=None,
                 comp_num=None, comp_icode=None,
                 is_hetatm=True, sep=':', parser=None):
        
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.comp_name = comp_name
        self.comp_num = comp_num
        
        self._pdb_id = pdb_id
        self._chain_id = chain_id
        self._comp_name = comp_name
        self._comp_num = comp_num
        self._comp_icode = comp_icode
        self.is_hetatm = is_hetatm
        self.sep = sep
        self.parser = parser

    @property
    def comp_icode(self):
        """str, read-only: the molecule insertion code."""
        if isinstance(self._comp_icode, str):
            return self._comp_icode
        else:
            return ' '

    def get_biopython_key(self, full_id=False):
        key = []
        if full_id:
            key = [self.pdb_id, 0, self.chain_id]

        if self.comp_name is not None and self.comp_num is not None:
            if self.comp_name == 'HOH' or self.comp_name == 'WAT':
                comp_id = ('W', self.comp_num, self.comp_icode)
            elif self.is_hetatm:
                comp_id = ('H_%s' % self.comp_name, self.comp_num,
                           self.comp_icode)
            else:
                comp_id = (' ', self.comp_num, self.comp_icode)

            if full_id:
                key.append(comp_id)
                return tuple(key)
            return comp_id

        if full_id:
            return tuple(key)

        return self.chain_id

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.to_string(self.sep))

    def to_string(self, sep=None):
        full_id = self.full_id

        # An entry object will always have a PDB and chain id.
        entry = list(full_id[0:2])

        # If it contains additional information about the molecule it
        # will also include them.
        if len(full_id) > 2:
            if full_id[2] is not None and full_id[3] is not None:
                comp_name = str(full_id[2]).strip()
                comp_num_and_icode = \
                    str(full_id[3]).strip() + str(full_id[4]).strip()
                entry += [comp_name, comp_num_and_icode]

        sep = sep or self.sep

        return sep.join(entry)

    @property
    def full_id(self):
        entry = [self.pdb_id, self.chain_id]
        if self.comp_name is not None and self.comp_num is not None:
            entry.append(self.comp_name)
            entry.append(self.comp_num)
            entry.append(self.comp_icode)
        return tuple(entry)