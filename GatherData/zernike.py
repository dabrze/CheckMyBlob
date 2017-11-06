import json
import math


class ZernikeCoefficient(object):
    def __init__(self, p, q, r, value_real, value_im):
        self.p = p
        self.q = q
        self.r = r
        self.value = complex(value_real, value_im)

    def __str__(self):
        return "p:{}, q:{}, r:{}, val:{}".format(self.p, self.q, self.r, self.value)

    def __repr__(self):
        return repr(self.__str__())

    @staticmethod
    def import_zernike_coefficient(path):
        cc = {}
        with open(path, 'r') as zernike_cc:
            data_cc = zernike_cc.read()
            for line in data_cc.splitlines():
                cc_row = json.loads(line)
                n = cc_row['n']
                li = cc_row['li']
                m = cc_row['m']

                element = ZernikeCoefficient(cc_row['p'], cc_row['q'], cc_row['r'], cc_row['cc_re'], cc_row['cc_im'])
                if (n,  li, m) in cc:
                    cc[(n,  li, m)].append(element)
                else:
                    cc[(n,  li, m)] = [element]
        return cc


class ZernikeMomentCache(object):
    def __init__(self, coefficients, moments, order=10):
        self.order = order
        self.coefficients = coefficients
        self.moments = moments
        self.zernike_moments = self.compute_moments(self.coefficients, moments)
        self.invariants = self.compute_invariants(self.zernike_moments)

    def compute_moments(self, coefficients, moments):
        zernike_moments = {}
        for n in range(self.order+1):
            li = 0
            for l in range(n % 2, n+1, 2):
                for m in range(l+1):
                    zm = complex(0.0, 0.0)

                    for cc in coefficients[(n,  li, m)]:
                        zm += cc.value.conjugate() * moments.get_moment(cc.p, cc.q, cc.r)

                    zm *= (3.0 / (4.0 * math.pi))
                    zernike_moments[(n, li, m)] = zm

                li += 1
        return zernike_moments

    @classmethod
    def get_moment(cls, n, l, m, zernike_moments):
        if m >= 0:
            return zernike_moments[(n, l//2, m)]
        sign = -1.0 if m%2 else 1.0
        return sign*zernike_moments[(n, l//2, abs(m))].conjugate()

    def compute_invariants(self, zernike_moments):
        invariants = {}
        for n in range(self.order+1):
            invariant = 0.0
            li = 0
            for l in range(n % 2, n+1, 2):
                for m in range(-l, l+1):
                    moment = self.get_moment(n, l, m, zernike_moments)
                    invariant += math.hypot(moment.real, moment.imag)

                invariants[(n, li)] = math.sqrt(invariant)
                li += 1
        return invariants

    def print_coeficient(self):
        for x, val in self.coefficents.iteritems():
            print x, val

    def print_invariants(self):
        for key in sorted(self.invariants.keys()):
            print key, self.invariants[key]
