import os
import re

from ignored_res import ELEMENTS_ELECTRONS

src_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dict', 'ligands.txt')


def is_int(s):
    return re.match(r"[-+]?\d+(\.0*)?$", s) is not None


class Formula(object):

    def __init__(self, formula):
        self.formula = formula

        atoms_list = self.formula.split(' ')
        self.charge = 0
        self.atoms = []
        self.count = []
        if len(atoms_list) > 0:
            if is_int(atoms_list[-1]):
                self.charge = int(atoms_list[-1])
                atoms_list = atoms_list[:-1]
            for elements_group in atoms_list:
                atom_count = self.get_atom_and_count(elements_group)
                self.atoms.append(atom_count[0])
                self.count.append(atom_count[1])

    def __str__(self):
        return "%s, atoms: %s, charge %d" % (self.formula, ';'.join(("%s %d" % (a, c) for a, c in zip(self.atoms, self.count))), self.charge)

    def get_atom_and_count(self, s):
        atom = 'C'
        count = 1
        atom_count = re.findall('[a-zA-Z]+|\\d+', s)
        if len(atom_count) > 0:
            atom = atom_count[0]
        if len(atom_count) > 1:
            count = int(atom_count[1])
        return atom, count

    def get_atoms_count(self):
        return sum(self.count)

    def get_non_h_atoms_count(self):
        return sum((c for a, c in zip(self.atoms, self.count) if a.strip().upper() != 'H'))

    def get_electron_count(self):
        #return self.charge+sum((ELEMENTS_ELECTRONS[a.strip().upper()]*c for a, c in zip(self.atoms, self.count)))
        # we should add charge for metals, otherwise we shouldn't
        return sum((ELEMENTS_ELECTRONS[a.strip().upper()]*c for a, c in zip(self.atoms, self.count)))

    def get_non_h_electron_count(self):
        #return self.charge+sum((ELEMENTS_ELECTRONS[a.strip().upper()]*c for a, c in zip(self.atoms, self.count) if a.strip().upper() != 'H'))
        # we should add charge for metals, otherwise we shouldn't
        return sum((ELEMENTS_ELECTRONS[a.strip().upper()]*c for a, c in zip(self.atoms, self.count) if a.strip().upper() != 'H'))

    def get_element_count(self, element='C'):
        element = element.strip().upper()
        return sum((c for a, c in zip(self.atoms, self.count) if a.strip().upper() == element))


def get_ligand_atoms_dict(filename=src_file):
    ligands = {}
    lig_file = open(filename, 'r')
    for line in lig_file.read().splitlines()[1:]:
        line_split = line.strip().split('\t')
        if len(line_split) > 3:
            res_name = line_split[0]
            count = line_split[1]
            name = line_split[2]
            formula = line_split[3]
            atoms_list = formula.split(' ')
            if len(atoms_list) > 0:
                form = Formula(formula)
            #atoms = form.get_non_h_atoms_count()
            #electrons = form.get_electron_count()
            #print res_name, atoms, electrons, form
            ligands[res_name] = (form, count, name)
    return ligands


def write_data(ligands, out_file):
    out = open(out_file, 'w')
    print >> out, ';'.join(("res_name", "count_in_pdb", "atoms_count", "non_h_atoms_count", "electron_count",
            "non_h_electron_count", "charge", "formula", "formula split", "name"))
    for res_name in ligands.iterkeys():
        form, count, name = ligands[res_name]
        print >> out, ";".join((
            res_name,
            count,
            str(form.get_atoms_count()),
            str(form.get_non_h_atoms_count()),
            str(form.get_electron_count()),
            str(form.get_non_h_electron_count()),
            str(form.charge),
            form.formula,
            '#'.join(("%s:%d" % (a, c) for a, c in zip(form.atoms, form.count))),
            name,
        ))
    out.close()


if __name__ == '__main__':
    ligands = get_ligand_atoms_dict(src_file)
    out_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dict', 'ligands.csv')
    write_data(ligands, out_file)
