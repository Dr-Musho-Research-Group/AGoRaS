
from math import gcd
import re
# folder = '500k_gen_august/first_half'
# os.chdir('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/'+folder)
# os.chdir('/.nfs/home/6/tempker/aae/generated_text/VAE_generated')
# PATH = "/.nfs/home/6/tempker/aae/generated_text/VAE_generated"
# EXT = "*.pkl"
# pkl_files = [file
#                  for path, subdir, files in os.walk(PATH)
#                  for file in glob(os.path.join(path, EXT))]
# def openf(pik):
#     try:
#         with open(pik, 'rb') as f:
#                 file = pk.load(f)
#                 return(file)

#     except:
#         pass
    
    
# file = [openf(x) for x in pkl_files]
# file = [x for x in file if x]
# Eqns = [x for y in file for x in y]
# Eqns = list(set(Eqns))
# Eqns = [x.strip() for x in Eqns]
# delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
#                '##', '..', '==', '# ', '((', '))', '++', ' - ','--','-=-']
# Eqns = [x for x in Eqns if all(i not in x for i in delete_list)]
# Eqns = [x.replace('   ',' + ') for x in Eqns]
# Eqns = [x.replace('  ',' ') for x in Eqns]



def get_subs(sub_re, split_re, molstr, mul=1):
    mol = {}
    for match in re.finditer(sub_re, molstr):
        n = int(match.group(2)) if match.group(2) else 1
        mol[match.group(1)] = mol.setdefault(match.group(1), 0) + (n * mul)
    for s in re.split(split_re, molstr):
        if s:
            mol[s] = mol.setdefault(s, 0) + mul
    return mol


def parse_molecule(molstr):
    elem_re = re.compile('([A-Z][a-z]?)(\d*)')
    elements = {}
    compounds = get_subs('\[([\w\(\)]~)\](\d*)', '\[[\w\(\)]~\]\d*', molstr)
    for s0, n0 in compounds.items():
        comps = get_subs('\(([\w\(\)]~)\)(\d*)', '\([\w\(\)]~\)\d*', s0, n0)
        for s1, n1 in comps.items():
            for elem_cnt in elem_re.finditer(s1):
                n = int(elem_cnt.group(2)) if elem_cnt.group(2) else 1
                elem = elem_cnt.group(1)
                elements[elem] = elements.setdefault(elem, 0) + (n * n1)
    return elements


def get_matrix(molecules, NR):
    elements = {}
    for idx, mol in enumerate(molecules):
        for elem, cnt in parse_molecule(mol).items():
            if elem not in elements:
                elements[elem] = [0] * len(molecules)
            elements[elem][idx] = cnt if idx < NR else -cnt
    return [row for row in elements.values()]

def lcm(list):
    if not list:
        return 1
    idx = next(i for i, x in enumerate(list) if x)
    n = list[idx]
    for m in list[(idx + 1):]:
        n *= m // gcd(n, m) if m else 1
    return abs(n)


def normalize(list):
    if len(list) < 2:
        return list
    g = list[0]
    for n in list[1:]:
        g = gcd(g, n)
    return [n // g for n in list]


def gauss_elim(matrix, NCOL):
    NROW = len(matrix)
    for k in range(min(NROW, NCOL)):
        cmax, imax = max((abs(matrix[i][k]), i) for i in range(k, NROW))
        matrix[k], matrix[imax] = matrix[imax], matrix[k]
        for below in range(k + 1, NROW):
            coeff, pivot = matrix[below][k], matrix[k][k]
            if not coeff:
                continue
            for col in range(k, NCOL):
                matrix[below][col] = ((pivot * matrix[below][col]) -
                                      (coeff * matrix[k][col]))
    return [row for row in matrix if any(row)]


def back_subst(matrix, known=None):
    if not matrix:
        return normalize(known)
    tosolve = matrix[-1][next(i for i, x in enumerate(matrix[-1]) if x):]
    if not known:
        known = (len(tosolve) - 1) * [1]
    t = lcm(tosolve) // lcm(tosolve[1:])
    known = [k * t for k in known]
    coeff = -sum([c * known[i] for i, c in enumerate(tosolve[1:])])
    return back_subst(matrix[:-1], [coeff // tosolve[0]] + known)


def balance_equation(skeleton):
    sides = [side.split('~') for side in skeleton.split('>')]
    NR = len(sides[0])
    molecules = [molstr.strip() for side in sides for molstr in side]
    matrix = get_matrix(molecules, NR)
    coeffs = back_subst(gauss_elim(matrix, len(molecules)))
    coeffstrs = [str(n) if n > 1 else '' for n in coeffs]
    molstrs = [''.join(t) for t in zip(coeffstrs, molecules)]
    return ' > '.join([' ~ '.join(molstrs[:NR]), ' ~ '.join(molstrs[NR:])])

# eqns = []
# for i in Eqns:
#     try:
#         b = balance_equation(i)
#         eqns.append(b)
#     except:
#         pass
    
