
ELEMENTS_ELECTRONS = {
    "H":1, "HE":2, "LI":3, "BE":4, "B":5, "C":6, "N":7, "O":8, "F":9, "NE":10, "NA":11, "MG":12,
    "AL":13, "SI":14, "P":15, "S":16, "CL":17, "AR":18, "K":19, "CA":20, "SC":21, "TI":22, "V":23, "CR":24,
    "MN":25, "FE":26, "CO":27, "NI":28, "CU":29, "ZN":30, "GA":31, "GE":32, "AS":33, "SE":34, "BR":35,
    "KR":36, "RB":37, "SR":38, "Y":39, "ZR":40, "NB":41, "MO":42, "TC":43, "RU":44, "RH":45, "PD":46,
    "AG":47, "CD":48, "IN":49, "SN":50, "SB":51, "TE":52, "I":53, "XE":54, "CS":55, "BA":56, "LA":57,
    "CE":58, "PR":59, "ND":60, "PM":61, "SM":62, "EU":63, "GD":64, "TB":65, "DY":66, "HO":67, "ER":68,
    "TM":69, "YB":70, "LU":71, "HF":72, "TA":73, "W":74, "RE":75, "OS":76, "IR":77, "PT":78, "AU":79,
    "HG":80, "TL":81, "PB":82, "BI":83, "PO":84, "AT":85, "RN":86, "FR":87, "RA":88, "AC":89, "TH":90,
    "PA":91, "U":92, "NP":93, "PU":94, "AM":95, "CM":96, "BK":97, "CF":98, "ES":99, "FM":100, "MD":101,
    "NO":102, "LR":103, "RF":104, "DB":105, "SG":106, "BH":107, "HS":108, "MT":109, "DS":110, "RG":111,
    'X': 6, 'D': 1
}

IGNORED_RESIDUES = set(
    (
        # PEPTIDE
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'CSH',
        'CYS',
        'GLN',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LEU',
        'LYS',
        'MET',
        'MSE',
        'ORN',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
        # DNA
        'DA',
        'DG',
        'DT',
        'DC',
        'A',
        'G',
        'T',
        'C',
        'U',
        #'YG',
        #'PSU',
        'I',
        # ELEMENTS
        'FE',    # ferrum
        'P',     # phosphorus
        'S',     # sulfur
        'I',     # iodin
        'BR',    # bromine
        'CL',    # chlorine
        'CA',    # calcium
        'CO',    # cobalt
        'CU',    # copper
        'ZN',    # zinc
        'MG',    # magnesium
        'MN',    # manganese
        'CD',    # cadmium
        'F',     # fluorine
        'NA',    # sodium
        'B',     # boron
        'HG',    # mercury
        'V',     # vanadium
        'PB',    # lead
        'HOH',   # water
        #'SO3',      #SULFITE ION
        #'SO4',      #sulphate-(SO4)
        #'PO3',      #PHOSPHITE ION
        #'PO4',      #phosphate-(PO4)
        #'CH2',      #Methylene
        #'SCN',      #THIOCYANATE ION
        #'CYN',      #CYANIDE ION
        #'CMO',      #CARBON MONOXIDE
        'GD',      #GADOLINIUM ATOM
        #'NH4',      #AMMONIUM ION
        #'NH2',      #AMINO GROUP
        'OH',      #HYDROXIDE ION
        'HO',      #HOLMIUM ATOM
        'DOD',      #DEUTERATED WATER
        'OXY',      #OXYGEN MOLECULE
        'C2O',      #CU-O-CU LINKAGE
        'C1O',      #CU-O LINKAGE
        'IN',      #INDIUM (III) ION
        'NI',      #NICKEL (II) ION
        #'NO3',      #NITRATE ION
        #'NO2',      #NITRITE ION
        'IOD',      #IODIDE ION
        'MTO',      #BOUND WATER
        'SR',      #STRONTIUM ION
        'YB',      #YTTERBIUM (III) ION
        'AL',      #ALUMINUM  ION
        'HYD',      #HYDROXY GROUP
        'IUM',      #URANYL(VI) ION
        'FLO',      #FLUORO GROUP
        'TE',      #te
        'K',      #POTASSIUM ION
        'LI',      #LITHIUM ION
        'RB',      #RUBIDIUM ION
        'FE2',      #FE(II) ION
        'NMO',      #NITROGEN MONOXIDE
        'OXO',      #OXO GROUP
        'CO2',      #CARBON DIOXIDE
        'BA',      #BARIUM ION
        'O',      #OXYGEN ATOM
        'PER',      #PEROXIDE ION
        'SM',      #SAMARIUM (III) ION
        'CS',      #CESIUM ION
        'MN3',      #MANGANESE (III) ION
        'CU1',      #COPPER (I) ION
        'H',      #HYDROGEN ATOM
        'TL',      #THALLIUM (I) ION
        'H2S',      #HYDROSULFURIC ACID
        'BRO',      #BROMO GROUP
        'IDO',      #IODO GROUP
        'PT',      #PLATINUM (II) ION
        'SI',      #.
        'GE',      #.
        'SN',      #.
        'BE',      #.
        'SC',      #.
        'Y',      #.
        'UR',      #.
        'CR',      #.
        'MO',      #.
        'W',      #.
        'AG',      #.
        'AU',      #.
        'AS',      #.
        'SE',      #.
        'HE',      #.
        'NE',      #.
        'AR',      #.
        'KR',      #.
        'XE',      #
        'GA',      #.
        'DUM',      #dummy atom
    )
)


KEEP_RESIDUES = set(
    (
        # PEPTIDE
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'CSH',
        'CYS',
        'GLN',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LEU',
        'LYS',
        'MET',
        'MSE',
        'ORN',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
        # DNA
        'DA',
        'DG',
        'DT',
        'DC',
        'A',
        'G',
        'T',
        'C',
        'U',
    )
)
