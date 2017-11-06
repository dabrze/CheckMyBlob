# coding: utf-8
__author__ = 'Marcin Kowiel'

import numpy as np
import math

EPS = 0.000000001


class MomentCache(object):
    def __init__(self, density, normalize):
        self.density = density

        self.normalize = normalize
        if self.normalize:
            self.normalize_const = self.get_normalization_const()
        else:
            self.normalize_const = None
        self.M = dict()

        self.M[(0, 0, 0)] = float(self.image_3d_moment(cm=(0, 0, 0), power=(0, 0, 0), normalize=normalize))
        if self.M[(0, 0, 0)] >= EPS:
            self.M[(1, 0, 0)] = self.image_3d_moment(cm=(0, 0, 0), power=(1, 0, 0), normalize=normalize) / self.M[(0, 0, 0)]
            self.M[(0, 1, 0)] = self.image_3d_moment(cm=(0, 0, 0), power=(0, 1, 0), normalize=normalize) / self.M[(0, 0, 0)]
            self.M[(0, 0, 1)] = self.image_3d_moment(cm=(0, 0, 0), power=(0, 0, 1), normalize=normalize) / self.M[(0, 0, 0)]

            self.x = {}
            self.y = {}
            self.z = {}

    def get_moment(self, p, q, r):
        if self.M[(0, 0, 0)] < EPS:
            return np.nan

        if (p,q,r) in self.M:
            return self.M[(p,q,r)]
        else:
            if p not in self.x:
                self.x[p] = self.get_vector(self.density.shape[0], self.M[(1, 0, 0)], power=p, normalize_const=self.normalize_const)
            if q not in self.y:
                self.y[q] = self.get_vector(self.density.shape[1], self.M[(0, 1, 0)], power=q, normalize_const=self.normalize_const)
            if r not in self.z:
                self.z[r] = self.get_vector(self.density.shape[2], self.M[(0, 0, 1)], power=r, normalize_const=self.normalize_const)

            self.M[(p, q, r)] = self.image_3d_moment(self.density, power=(p, q, r), x=self.x[p], y=self.y[q], z=self.z[r])

            return self.M[(p, q, r)]

    def get_normalization_const(self):
        M000 = float(self.image_3d_moment(cm=(0, 0, 0), power=(0, 0, 0), normalize=False))
        if M000 >= EPS:
            M100 = self.image_3d_moment(cm=(0, 0, 0), power=(1, 0, 0), normalize=False) / M000
            M010 = self.image_3d_moment(cm=(0, 0, 0), power=(0, 1, 0), normalize=False) / M000
            M001 = self.image_3d_moment(cm=(0, 0, 0), power=(0, 0, 1), normalize=False) / M000
            xx, yy, zz = self.get_index_minus_cm(cm=(M100, M010, M001), power=(2,2,2), x=None, y=None, z=None, normalize=False)
            dist = xx+yy+zz
            dist = dist*(self.density>0)
            max_len = np.sqrt(dist.max())
            #print 'Norm', self.density.shape, 0.5*max(self.density.shape)*math.sqrt(3), max_len
            return max_len
        else:
           longest = max(self.density.shape)
           return 0.5*longest*math.sqrt(3)

    @staticmethod
    def get_vector(length, cm, power=1, normalize_const=None):
        if power == 1:
            if normalize_const is not None:
                return ((1.0/normalize_const)*np.arange(length))-float(cm)
            return np.arange(length)-float(cm)
        if normalize_const is not None:
            return np.power(((1.0/normalize_const)*np.arange(length))-float(cm), power)
        return np.power(np.arange(length)-float(cm), power)

    def get_index_minus_cm(self, cm=(0, 0, 0), power=(1,1,1), x=None, y=None, z=None, normalize=False):
        len_x, len_y, len_z = self.density.shape

        if normalize is True:
            normalize_const = self.get_normalization_const()
        else:
            normalize_const = None

        if x is None:
            x = self.get_vector(len_x, cm[0], power[0], normalize_const=normalize_const)
        if y is None:
            y = self.get_vector(len_y, cm[1], power[1], normalize_const=normalize_const)
        if z is None:
            z = self.get_vector(len_z, cm[2], power[2], normalize_const=normalize_const)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return xx, yy, zz

    def image_3d_moment(self, cm=(0, 0, 0), power=(0, 0, 0), x=None, y=None, z=None, normalize=False):
        if power == (0, 0, 0):
            return np.sum(self.density)

        xx, yy, zz = self.get_index_minus_cm(cm=cm, power=power, x=x, y=y, z=z, normalize=normalize)

        if power[0] == 0 and power[1] == 0:
            return np.sum(zz*self.density)
        if power[0] == 0 and power[2] == 0:
            return np.sum(yy*self.density)
        if power[1] == 0 and power[2] == 0:
            return np.sum(xx*self.density)

        if power[0] == 0:
            return np.sum(yy*zz*self.density)
        if power[1] == 0:
            return np.sum(xx*zz*self.density)
        if power[2] == 0:
            return np.sum(xx*yy*self.density)

        return np.sum(xx*yy*zz*self.density)


class GeometricalInvariantCache(object):
    def __init__(self, moments):
        self.moments = moments
        self.invariants = self.compute_invariants()

    def _compute_ci(self):
        a1 = self.moments.get_moment(0, 0, 2) - self.moments.get_moment(0, 2, 0)
        a2 = self.moments.get_moment(0, 2, 0) - self.moments.get_moment(2, 0, 0)
        a3 = self.moments.get_moment(2, 0, 0) - self.moments.get_moment(0, 0, 2)

        b1 = self.moments.get_moment(0, 2, 1) - self.moments.get_moment(2, 0, 1)
        b2 = self.moments.get_moment(1, 0, 2) - self.moments.get_moment(1, 2, 0)
        b3 = self.moments.get_moment(2, 1, 0) - self.moments.get_moment(0, 1, 2)
        b4 = self.moments.get_moment(0, 0, 3) - self.moments.get_moment(2, 0, 1) - 2 * self.moments.get_moment(0, 2, 1)
        b5 = self.moments.get_moment(0, 3, 0) - self.moments.get_moment(0, 2, 1) - 2 * self.moments.get_moment(2, 1, 0)
        b6 = self.moments.get_moment(3, 0, 0) - self.moments.get_moment(1, 2, 0) - 2 * self.moments.get_moment(1, 0, 2)
        b7 = self.moments.get_moment(0, 2, 1) - self.moments.get_moment(0, 0, 3) + 2 * self.moments.get_moment(2, 0, 1)
        b8 = self.moments.get_moment(1, 0, 2) - self.moments.get_moment(3, 0, 0) + 2 * self.moments.get_moment(1, 2, 0)
        b9 = self.moments.get_moment(2, 1, 0) - self.moments.get_moment(0, 3, 0) + 2 * self.moments.get_moment(0, 1, 2)
        b10 = self.moments.get_moment(0, 2, 1) + self.moments.get_moment(2, 0, 1) - 3 * self.moments.get_moment(0, 0, 3)
        b11 = self.moments.get_moment(0, 1, 2) + self.moments.get_moment(2, 1, 0) - 3 * self.moments.get_moment(0, 3, 0)
        b12 = self.moments.get_moment(1, 0, 2) + self.moments.get_moment(1, 2, 0) - 3 * self.moments.get_moment(3, 0, 0)
        b13 = self.moments.get_moment(0, 2, 1) + self.moments.get_moment(0, 0, 3) + 3 * self.moments.get_moment(2, 0, 1)
        b14 = self.moments.get_moment(1, 0, 2) + self.moments.get_moment(3, 0, 0) + 3 * self.moments.get_moment(1, 2, 0)
        b15 = self.moments.get_moment(2, 1, 0) + self.moments.get_moment(0, 3, 0) + 3 * self.moments.get_moment(0, 1, 2)
        b16 = self.moments.get_moment(0, 1, 2) + self.moments.get_moment(0, 3, 0) + 3 * self.moments.get_moment(2, 1, 0)
        b17 = self.moments.get_moment(2, 0, 1) + self.moments.get_moment(0, 0, 3) + 3 * self.moments.get_moment(0, 2, 1)
        b18 = self.moments.get_moment(1, 2, 0) + self.moments.get_moment(3, 0, 0) + 3 * self.moments.get_moment(1, 0, 2)

        g1 = self.moments.get_moment(0, 2, 2) - self.moments.get_moment(4, 0, 0)
        g2 = self.moments.get_moment(2, 0, 2) - self.moments.get_moment(0, 4, 0)
        g3 = self.moments.get_moment(2, 2, 0) - self.moments.get_moment(0, 0, 4)
        g4 = self.moments.get_moment(1, 1, 2) + self.moments.get_moment(1, 3, 0) + self.moments.get_moment(3, 1, 0)
        g5 = self.moments.get_moment(1, 2, 1) + self.moments.get_moment(1, 0, 3) + self.moments.get_moment(3, 0, 1)
        g6 = self.moments.get_moment(2, 1, 1) + self.moments.get_moment(0, 1, 3) + self.moments.get_moment(0, 3, 1)
        g7 = self.moments.get_moment(0, 2, 2) - self.moments.get_moment(2, 2, 0) + self.moments.get_moment(0, 0, 4) - self.moments.get_moment(4, 0, 0)
        g8 = self.moments.get_moment(2, 0, 2) - self.moments.get_moment(0, 2, 2) + self.moments.get_moment(4, 0, 0) - self.moments.get_moment(0, 4, 0)
        g9 = self.moments.get_moment(2, 2, 0) - self.moments.get_moment(2, 0, 2) + self.moments.get_moment(0, 4, 0) - self.moments.get_moment(0, 0, 4)

        rgyr = math.sqrt((self.moments.get_moment(2, 0, 0) + self.moments.get_moment(0, 2, 0) + self.moments.get_moment(0, 0, 2))/(3*self.moments.get_moment(0, 0, 0)))
        s3 = 1/(self.moments.get_moment(0, 0, 0)**3 * rgyr**9)
        s4 = 1/(self.moments.get_moment(0, 0, 0)**4 * rgyr**9)

        ci = 4*s3*(
                self.moments.get_moment(1, 1, 0) * (self.moments.get_moment(0, 2, 1) * (3*g2 - 2*g3 - g1)
                - self.moments.get_moment(2, 0, 1) * (3*g1 - 2*g3 - g2) + b12*g5
                - b11*g6 + self.moments.get_moment(0, 0, 3)*g8) + self.moments.get_moment(1, 0, 1) * (self.moments.get_moment(2, 1, 0) * (3*g1 - 2*g2 - g3)
                - self.moments.get_moment(0, 1, 2) * (3*g3 - 2*g2 - g1) + b10*g6 - b12*g4
                + self.moments.get_moment(0, 3, 0)*g7) + self.moments.get_moment(0, 1, 1) * (self.moments.get_moment(1, 0, 2) * (3*g3 - 2*g1 - g2)
                - self.moments.get_moment(1, 2, 0) * (3*g2 - 2*g1 - g3)
                + b11*g4 - b10*g5 + self.moments.get_moment(3, 0, 0)*g9)
                + self.moments.get_moment(0, 0, 2) * (b18*g6 - b15*g5 - 2 * (self.moments.get_moment(1, 1, 1)*g8 + b1*g4))
                + self.moments.get_moment(0, 2, 0) * (b17*g4 - b14*g6 - 2 * (self.moments.get_moment(1, 1, 1)*g7 + b3*g5))
                + self.moments.get_moment(2, 0, 0) * (b16*g5 - b13*g4 - 2 * (self.moments.get_moment(1, 1, 1)*g9 + b2*g6))
            ) - 16*s4*(
                self.moments.get_moment(0, 1, 1) * a2 * a3 * b2 + self.moments.get_moment(1, 0, 1) * a1 * a2 * b3
                + self.moments.get_moment(1, 1, 0) * a1 * a3 * b1 - self.moments.get_moment(1, 1, 1) * a1 * a2 * a3
                - self.moments.get_moment(0, 1, 1) * self.moments.get_moment(0, 1, 1) * (self.moments.get_moment(1, 1, 1) * a1 - self.moments.get_moment(0, 1, 1) * b2 - self.moments.get_moment(1, 0, 1) * b5 - self.moments.get_moment(1, 1, 0) * b7)
                - self.moments.get_moment(1, 0, 1) * self.moments.get_moment(1, 0, 1) * (self.moments.get_moment(1, 1, 1) * a3 - self.moments.get_moment(1, 0, 1) * b3 - self.moments.get_moment(1, 1, 0) * b4 - self.moments.get_moment(0, 1, 1) * b8)
                - self.moments.get_moment(1, 1, 0) * self.moments.get_moment(1, 1, 0) * (self.moments.get_moment(1, 1, 1) * a2 - self.moments.get_moment(1, 1, 0) * b1 - self.moments.get_moment(0, 1, 1) * b6 - self.moments.get_moment(1, 0, 1) * b9)
                + self.moments.get_moment(0, 1, 1) * self.moments.get_moment(1, 0, 1) * (self.moments.get_moment(0, 0, 2) * b1 + self.moments.get_moment(0, 2, 0) * b4 + self.moments.get_moment(2, 0, 0) * b7)
                + self.moments.get_moment(0, 1, 1) * self.moments.get_moment(1, 1, 0) * (self.moments.get_moment(0, 2, 0) * b3 + self.moments.get_moment(2, 0, 0) * b5 + self.moments.get_moment(0, 0, 2) * b9)
                + self.moments.get_moment(1, 0, 1) * self.moments.get_moment(1, 1, 0) * (self.moments.get_moment(2, 0, 0) * b2 + self.moments.get_moment(0, 0, 2) * b6 + self.moments.get_moment(0, 2, 0) * b8)
            )

        return ci

    def compute_invariants(self):
        invariants = {}
        keys = [
            'O3', 'O4', 'O5', 'FL',
            'O3_norm', 'O4_norm', 'O5_norm', 'FL_norm',
            'I1', 'I2', 'I3', 'I4', 'I5', 'I6',
            'I1_norm', 'I2_norm', 'I3_norm', 'I4_norm', 'I5_norm', 'I6_norm',
            'M000',
            'E1', 'E2', 'E3', 'E3_E1', 'E2_E1', 'E3_E2',
            'CI',
        ]
        for key in keys:
            invariants[key] = np.nan

        M000 = self.moments.get_moment(0,0,0)
        if M000 < EPS or np.isnan(M000):
            return invariants

        M110 = self.moments.get_moment(1, 1, 0)
        M101 = self.moments.get_moment(1, 0 ,1)
        M011 = self.moments.get_moment(0, 1, 1)

        M200 = self.moments.get_moment(2, 0, 0)
        M020 = self.moments.get_moment(0, 2, 0)
        M002 = self.moments.get_moment(0, 0, 2)

        M111 = self.moments.get_moment(1, 1, 1)

        M120 = self.moments.get_moment(1, 2, 0)
        M102 = self.moments.get_moment(1, 0, 2)
        M210 = self.moments.get_moment(2, 1, 0)
        M201 = self.moments.get_moment(2, 0, 1)
        M021 = self.moments.get_moment(0, 2, 1)
        M012 = self.moments.get_moment(0, 1, 2)

        M300 = self.moments.get_moment(3, 0, 0)
        M030 = self.moments.get_moment(0, 3, 0)
        M003 = self.moments.get_moment(0, 0, 3)

        M220 = self.moments.get_moment(2, 2, 0)
        M202 = self.moments.get_moment(2, 0, 2)
        M022 = self.moments.get_moment(0, 2, 2)

        M400 = self.moments.get_moment(4, 0, 0)
        M040 = self.moments.get_moment(0, 4, 0)
        M004 = self.moments.get_moment(0, 0, 4)

        M130 = self.moments.get_moment(1, 3, 0)
        M103 = self.moments.get_moment(1, 0, 3)
        M310 = self.moments.get_moment(3, 1, 0)
        M301 = self.moments.get_moment(3, 0, 1)
        M031 = self.moments.get_moment(0, 3, 1)
        M013 = self.moments.get_moment(0, 1, 3)

        M211 = self.moments.get_moment(2, 1, 1)
        M121 = self.moments.get_moment(1, 2, 1)
        M112 = self.moments.get_moment(1, 1, 2)

        covariance = np.array([M200, M110, M101, M110, M020, M011, M101, M011, M002])
        covariance.shape = (3, 3)
        covariance = covariance/M000
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        eigenvalues_sort = sorted([abs(e) for e in eigenvalues])

        # something is wrong with normalisation constants!
        # check it
        invariants['O3'] = (M200+M020+M002) # / M000^((2+0+0)/3+1)
        invariants['O4'] = (M200*M020+M200*M002+M020*M002-M110*M110-M101*M101-M011*M011) # / M000^(2*(2+0+0+3)/3)
        invariants['O5'] = (M200*M020*M002+2*M110*M101*M011-M200*M011*M011-M020*M101*M101-M002*M110*M110)# / M000^(3*(2+0+0+3)/3) /(M000_sq*M000_sq*M000)
        invariants['FL'] = (M300*M300+M030*M030+M003*M003+6*(M120*M120+M102*M102+M210*M210+M201*M201+M021*M021+M012*M012) +
                         15*M111-3*(M300*(M120+M102)+M030*(M210+M012)+M003*(M201+M021)+M120*M102+M210*M012+M201*M021)) # M000^(2*(3+0+0+3)/3) #/(M000_sq*M000_sq)
        invariants['O3_norm'] = invariants['O3'] / math.pow(M000, 5.0/3.0)
        invariants['O4_norm'] = invariants['O4'] / math.pow(M000, 10.0/3.0)
        invariants['O5_norm'] = invariants['O5'] / math.pow(M000, 5)
        invariants['FL_norm'] = invariants['FL'] / math.pow(M000, 4)
        # from 3-D Surface Moment Invariants Dong Xu and Hua Li 2006
        invariants['I1'] = (M400+M040+M004+2*(M220+M202+M022))
        invariants['I2'] = (M400*M040+M400*M004+M040*M004 +
                         3*(M220*M220+M202*M202+M022*M022) -
                         4*(M103*M301+M130*M310) +
                         2*(M220*M202+M220*M022+M202*M022) +
                         2*(M400*M022+M040*M202+M004*M220) -
                         4*(M103*M121+M130*M112+M013*M211+M121*M301+M112*M310+M211*M031) +
                         4*(M211*M211+M121*M121+M112*M112))
        invariants['I3'] = (M400*M400+M040*M040+M004*M004+
                         4*(M130*M130+M103*M103+M310*M310+M301*M301+M031*M031+M013*M013) +
                         6*(M220*M220+M202*M202+M022*M022) +
                         12*(M211*M211+M121*M121+M112*M112))
        invariants['I4'] = (M300*M300+M030*M030+M003*M003+6*M111 +
                         3*(M120*M120+M102*M102+M210*M210+M201*M201+M021*M021+M012*M012))
        invariants['I5'] = (M300*M300+M030*M030+M003*M003+(M120*M120+M102*M102+M210*M210+M201*M201+M021*M021+M012*M012) +
                         2*(M300*(M120+M102)+M030*(M210+M012)+M003*(M201+M021)+M120*M102+M210*M012+M201*M021))
        invariants['I6'] = (M200*(M400+M220+M202)+M020*(M040+M220+M022)+M002*(M004+M202+M022) +
                         M110*(M310+M130+M112)+M101*(M301+M121+M103)+M011*(M211+M031+M013))

        invariants['I1_norm'] = invariants['I1'] / math.pow(M000, 7.0/3.0)
        invariants['I2_norm'] = invariants['I2'] / math.pow(M000, 14.0/3.0)
        invariants['I3_norm'] = invariants['I3'] / math.pow(M000, 14.0/3.0)
        invariants['I4_norm'] = invariants['I4'] / math.pow(M000, 12/3)
        invariants['I5_norm'] = invariants['I5'] / math.pow(M000, 12/3)
        invariants['I6_norm'] = invariants['I6'] / math.pow(M000, 12/3)

        invariants['M000'] = M000

        invariants['E1'] = eigenvalues_sort[2]
        invariants['E2'] = eigenvalues_sort[1]
        invariants['E3'] = eigenvalues_sort[0]
        invariants['E2_E1'] = eigenvalues_sort[1]/eigenvalues_sort[2]
        invariants['E3_E1'] = eigenvalues_sort[0]/eigenvalues_sort[2]
        invariants['E3_E2'] = eigenvalues_sort[0]/eigenvalues_sort[1]

        invariants['CI'] = self._compute_ci()

        return invariants
