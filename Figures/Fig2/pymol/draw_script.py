from pymol.cgo import *
from pymol import cmd

def signOfFloat(f):
        if f < 0: return -1
        if f > 0: return 1
        return 0

def sqC(v, n):
        return signOfFloat(math.cos(v)) *  math.pow(math.fabs(math.cos(v)), n)

def sqCT(v, n, alpha):
        return alpha + sqC(v, n)

def sqS(v, n):
        return signOfFloat(math.sin(v)) * math.pow(math.fabs(math.sin(v)), n)

def sqEllipsoid(x, y, z, a1, a2, a3, u, v, n, e):
        x = a1 * sqC(u, n) * sqC(v, e) + x
        y = a2 * sqC(u, n) * sqS(v, e) + y
        z = a3 * sqS(u, n) + z
        nx = sqC(u, 2 - n) * sqC(v, 2 - e) / a1
        ny = sqC(u, 2 - n) * sqS(v, 2 - e) / a2
        nz = sqS(u, 2 - n) / a3
        return x, y, z, nx, ny, nz

def sqToroid(x, y, z, a1, a2, a3, u, v, n, e, alpha):
        a1prime = 1.0 / (a1 + alpha)
        a2prime = 1.0 / (a2 + alpha)
        a3prime = 1.0 / (a3 + alpha)
        x = a1prime * sqCT(u, e, alpha) * sqC(v, n)
        y = a2prime * sqCT(u, e, alpha) * sqS(v, n)
        z = a3prime * sqS(u, e)
        nx = sqC(u, 2 - e) * sqC(v, 2 - n) / a1prime
        ny = sqC(u, 2 - e) * sqS(v, 2 - n) / a2prime
        nz = sqS(u, 2 - e) / a3prime
        return x, y, z, nx, ny, nz

def makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, n, e, u1, u2, v1, v2, u_segs, v_segs, color=[0.5, 0.5, 0.5]):

        r, g, b = color

        # Calculate delta variables */
        dU = (u2 - u1) / u_segs
        dV = (v2 - v1) / v_segs

        o = [ BEGIN, TRIANGLES ]

        U = u1
        for Y in range(0, u_segs):
                # Initialize variables for loop */
                V = v1
                for X in range(0, v_segs):
                        # VERTEX #1 */
                        x1, y1, z1, n1x, n1y, n1z = sqEllipsoid(x, y, z, a1, a2, a3, U, V, n, e)
                        x2, y2, z2, n2x, n2y, n2z = sqEllipsoid(x, y, z, a1, a2, a3, U + dU, V, n, e)
                        x3, y3, z3, n3x, n3y, n3z = sqEllipsoid(x, y, z, a1, a2, a3, U + dU, V + dV, n, e)
                        x4, y4, z4, n4x, n4y, n4z = sqEllipsoid(x, y, z, a1, a2, a3, U, V + dV, n, e)

                        o.extend([COLOR, r, g, b, NORMAL, n1x, n1y, n1z, VERTEX, x1, y1, z1])
                        o.extend([COLOR, r, g, b, NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
                        o.extend([COLOR, r, g, b, NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])
                        o.extend([COLOR, r, g, b, NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
                        o.extend([COLOR, r, g, b, NORMAL, n3x, n3y, n3z, VERTEX, x3, y3, z3])
                        o.extend([COLOR, r, g, b, NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])

                        # Update variables for next loop */
                        V += dV
                # Update variables for next loop */
                U += dU
        o.append(END)
        return o

def makeSuperQuadricToroid(x, y, z, a1, a2, a3, alpha, n, e, u1, u2, v1, v2, u_segs, v_segs, color=[0.5, 0.5, 0.5]):

        r, g, b = color

        # Calculate delta variables */
        dU = (u2 - u1) / u_segs
        dV = (v2 - v1) / v_segs

        o = [ BEGIN, TRIANGLES ]

        U = u1
        for Y in range(0, u_segs):
                # Initialize variables for loop */
                V = v1
                for X in range(0, v_segs):
                        # VERTEX #1 */
                        x1, y1, z1, n1x, n1y, n1z = sqToroid(x, y, z, a1, a2, a3, U, V, n, e, alpha)
                        x2, y2, z2, n2x, n2y, n2z = sqToroid(x, y, z, a1, a2, a3, U + dU, V, n, e, alpha)
                        x3, y3, z3, n3x, n3y, n3z = sqToroid(x, y, z, a1, a2, a3, U + dU, V + dV, n, e, alpha)
                        x4, y4, z4, n4x, n4y, n4z = sqToroid(x, y, z, a1, a2, a3, U, V + dV, n, e, alpha)

                        o.extend([COLOR, r, g, b, NORMAL, n1x, n1y, n1z, VERTEX, x1, y1, z1])
                        o.extend([COLOR, r, g, b, NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
                        o.extend([COLOR, r, g, b, NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])
                        o.extend([COLOR, r, g, b, NORMAL, n2x, n2y, n2z, VERTEX, x2, y2, z2])
                        o.extend([COLOR, r, g, b, NORMAL, n3x, n3y, n3z, VERTEX, x3, y3, z3])
                        o.extend([COLOR, r, g, b, NORMAL, n4x, n4y, n4z, VERTEX, x4, y4, z4])

                        # Update variables for next loop */
                        V += dV
                # Update variables for next loop */
                U += dU
        o.append(END)
        return o

def makeEllipsoid(x, y, z, a1, a2, a3):
                return makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, 1.0, 1.0, -math.pi / 2, math.pi / 2, -math.pi, math.pi, 10, 10)

def makeCylinder(x, y, z, a1, a2, a3):
                return makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, 0.0, 1.0, -math.pi / 2, math.pi / 2, -math.pi, math.pi, 10, 10)

def makeSpindle(x, y, z, a1, a2, a3):
                return makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, 2.0, 1.0, -math.pi / 2, math.pi / 2, -math.pi, math.pi, 10, 10)

def makeDoublePyramid(x, y, z, a1, a2, a3):
                return makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, 2.0, 2.0, -math.pi / 2, math.pi / 2, -math.pi, math.pi, 10, 10)

def makePillow(x, y, z, a1, a2, a3):
                return makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, 1.0, 0.0, -math.pi, math.pi, -math.pi, math.pi, 10, 10)

def makeRoundCube(x, y, z, a1, a2, a3):
                return makeSuperQuadricEllipsoid(x, y, z, a1, a2, a3, 0.2, 0.2, -math.pi / 2, math.pi / 2, -math.pi, math.pi, 10, 10)

def makeToroid(x, y, z, a1, a2, a3, alpha):
                return makeSuperQuadricToroid(x, y, z, a1, a2, a3, alpha, 1.0, 1.0, -math.pi, math.pi, -math.pi, math.pi, 10, 10)

def makeSpheres(xyz_list, r=0.3, color=[1.0, 0.0, 0.0]):
    out = []
    out.extend([COLOR, color[0], color[1], color[2]])
    for point in xyz_list:
        out.extend([SPHERE, point[0], point[1], point[2], r])
    return out

def length_sq(p1, p2):
    a = p1[0]-p2[0]
    b = p1[1]-p2[1]
    c = p1[2]-p2[2]
    return a*a+b*b+c*c

def makeBonds(xyz_list, bond_dist=2.15, r=0.3, color=[1.0, 0.0, 0.0]):
    out = []
    out.extend([COLOR, color[0], color[1], color[2]])
    for point1 in xyz_list:
        for point2 in xyz_list:
            if length_sq(point1, point2) < bond_dist*bond_dist:
                #CYLINDER, x1, y1, z1, x2, y2, z2, radius, r1, g1, b1, r2, g2, b2
                out.extend([CYLINDER, point1[0], point1[1], point1[2], point2[0], point2[1], point2[2], r, color[0], color[1], color[2], color[0], color[1], color[2]])
    return out

brightorange = [1.0, 0.7, 0.2]
tv_yellow = [1.0, 1.0, 0.2]
oxygen = [1.0, 0.3, 0.3]

points_3wh1 = [[20.617,12.842,1.043],[20.535,12.842,1.217],[20.894,12.842,0.869],[20.645,12.842,1.391],[21.170,12.648,0.695],[21.447,12.453,0.521],[21.723,12.259,0.348],[21.834,12.259,0.521],[21.613,12.453,0.174],[22.028,11.870,0.521],[22.028,12.064,0.521],[21.695,12.648,0.000],[22.028,11.675,0.521],[21.584,12.648,-0.174],[22.222,11.480,0.521],[21.667,12.648,-0.348],[22.333,11.286,0.695],[22.526,11.286,0.695],[21.750,12.648,-0.521],[22.803,11.091,0.521],[21.639,12.842,-0.695],[23.079,11.091,0.348],[23.356,10.897,0.174],[23.522,10.508,-0.174],[23.632,10.702,0.000],[23.522,10.313,-0.174],[23.743,10.897,0.174],[23.522,10.118,-0.174],[23.937,10.897,0.174],[23.522,9.924,-0.174],[24.131,10.897,0.174],[23.411,9.729,-0.348],[24.242,11.091,0.348],[23.411,9.535,-0.348],[24.435,11.091,0.348],[22.719,9.340,-0.521],[22.912,9.340,-0.521],[23.023,9.340,-0.348],[23.217,9.340,-0.348],[24.629,11.091,0.348],[22.608,9.145,-0.695],[23.411,9.145,-0.348],[24.823,11.091,0.348],[22.414,9.145,-0.695],[23.687,8.951,-0.521],[24.934,10.897,0.521],[24.934,11.091,0.521],[22.303,9.145,-0.869],[23.964,8.756,-0.695],[25.210,10.702,0.348],[25.045,11.286,0.695],[25.045,11.480,0.695],[25.045,11.675,0.695],[25.045,11.870,0.695],[25.127,12.064,0.521],[22.192,9.145,-1.043],[24.240,8.562,-0.869],[24.628,8.562,-0.869],[24.738,8.756,-0.695],[25.487,10.702,0.174],[25.127,12.259,0.521],[24.434,8.367,-0.869],[24.932,8.756,-0.695],[25.652,9.729,-0.174],[25.763,9.924,0.000],[25.763,10.118,0.000],[25.846,10.313,-0.174],[25.763,10.508,0.000],[25.045,12.453,0.695],[25.155,12.453,0.869],[25.266,12.453,1.043],[24.434,8.173,-0.869],[25.126,8.756,-0.695],[25.652,9.535,-0.174],[25.377,12.648,1.217],[24.434,7.978,-0.869],[25.237,8.756,-0.521],[25.652,9.340,-0.174],[25.488,12.842,1.391],[24.434,7.783,-0.869],[25.430,8.756,-0.521],[25.652,9.145,-0.174],[25.599,12.842,1.564],[24.434,7.589,-0.869],[25.541,8.951,-0.348],[25.846,8.951,-0.174],[25.710,12.842,1.738],[24.323,7.394,-1.043],[25.929,8.756,-0.348],[26.205,8.756,-0.521],[25.904,13.037,1.738],[24.240,7.200,-0.869],[24.351,7.200,-0.695],[26.482,8.562,-0.695],[26.097,13.232,1.738],[26.758,8.367,-0.869],[26.952,8.562,-0.869],[27.063,8.562,-0.695],[26.291,13.232,1.738],[27.174,8.562,-0.521],[26.485,13.232,1.738],[27.368,8.562,-0.521],[26.679,13.232,1.738],[27.561,8.562,-0.521],[26.872,13.232,1.738],[27.672,8.562,-0.348],[27.149,13.232,1.564],[27.866,8.562,-0.348],[27.591,12.648,1.043],[27.508,12.842,1.217],[27.508,13.037,1.217],[27.425,13.232,1.391],[28.060,8.756,-0.348],[27.867,12.453,0.869],[28.171,8.756,-0.174],[27.702,12.259,1.217],[27.978,12.259,1.043],[27.813,12.064,1.391],[27.924,12.064,1.564],[26.572,22.377,5.215],[26.849,21.988,5.041],[26.766,22.183,5.215],[27.042,21.793,5.041],[27.236,21.599,5.041],[27.430,21.404,5.041],[27.347,21.015,5.215],[27.347,21.210,5.215],[27.264,20.820,5.389],[27.623,21.404,5.041],[27.264,20.626,5.389],[27.900,21.599,4.867],[27.264,20.431,5.389],[28.176,21.599,4.693],[26.988,20.237,5.563],[28.453,21.599,4.520],[27.181,20.042,5.563],[28.564,21.599,4.693],[27.375,19.848,5.563],[27.375,19.653,5.563],[27.569,19.458,5.563],[27.762,19.264,5.563],[27.845,19.069,5.389],[27.873,19.069,5.736],[28.067,18.875,5.736],[27.984,18.680,5.910],[28.095,18.680,6.084],[28.206,18.680,6.258],[28.123,18.680,6.432],[28.234,18.485,6.606],[28.151,18.485,6.779],[28.262,17.902,6.953],[28.262,18.096,6.953],[28.262,18.291,6.953],[28.262,18.485,6.953],[28.262,17.707,6.953],[28.373,18.680,7.127],[28.262,17.512,6.953],[28.290,18.875,7.301],[28.484,19.069,7.301],[28.373,17.318,7.127],[28.595,19.264,7.475],[28.789,19.458,7.475],[28.262,17.123,6.953],[28.900,19.458,7.648],[28.069,16.929,6.953],[29.011,19.264,7.822],[27.958,16.734,6.779],[28.928,19.264,7.996],[27.958,16.540,6.779],[28.845,19.458,8.170],[27.847,16.345,6.606],[28.762,19.458,8.344],[27.459,16.150,6.606],[27.653,16.150,6.606],[28.679,19.458,8.518],[27.348,16.150,6.432],[28.596,19.458,8.691],[28.514,19.458,8.865],[28.625,19.653,9.039],[28.625,19.848,9.039],[28.735,19.458,9.213],[28.735,20.042,9.213],[28.846,19.264,9.387],[28.653,20.237,9.387],[29.040,19.069,9.387],[29.234,18.875,9.387],[28.653,20.431,9.387],[29.345,18.875,9.561],[28.653,20.626,9.387],[29.538,18.875,9.561],[28.764,20.820,9.561],[28.792,20.820,9.908],[29.732,18.291,9.561],[29.732,18.680,9.561],[28.874,21.015,9.734],[28.709,20.820,10.082],[29.732,18.096,9.561],[29.843,18.485,9.734],[29.677,18.680,10.082],[28.626,20.820,10.256],[29.732,17.902,9.561],[29.954,18.680,9.908],[29.788,18.680,10.256],[28.543,21.015,10.430],[29.732,17.707,9.561],[29.706,18.680,10.430],[28.654,21.015,10.604],[29.732,17.512,9.561],[29.816,18.680,10.604],[28.571,21.015,10.777],[29.732,17.318,9.561],[29.734,18.680,10.777],[28.682,21.015,10.951],[29.510,17.123,9.213],[29.621,17.123,9.387],[29.651,18.875,10.951],[28.793,21.015,11.125],[29.762,18.875,11.125],[28.904,20.820,11.299],[29.098,20.626,11.299],[29.292,20.626,11.299],[29.873,18.875,11.299],[29.402,20.431,11.473],[29.596,20.237,11.473],[29.984,18.680,11.473],[29.707,20.237,11.647],[30.094,18.485,11.647],[30.288,18.485,11.647],[30.399,18.291,11.820],[30.593,18.096,11.820],[30.704,18.096,11.994],[30.815,17.902,12.168],[30.732,17.513,12.342],[30.732,17.707,12.342],[30.649,17.123,12.516],[30.649,17.318,12.516],[30.455,16.929,12.516],[30.179,16.929,12.690],[29.902,16.734,12.863],[29.321,16.540,12.863],[29.515,16.540,12.863],[29.709,16.540,12.863],[26.985,18.291,3.129],[27.096,18.291,3.303],[27.206,18.291,3.477],[27.317,18.096,3.650],[27.428,18.096,3.824],[27.345,18.096,3.998],[27.263,18.096,4.172],[27.180,18.096,4.346],[27.291,17.902,4.520],[27.208,17.707,4.693],[27.125,17.318,4.867],[27.125,17.512,4.867],[27.125,16.929,4.867],[27.125,17.123,4.867],[27.319,16.734,4.867],[27.430,16.734,5.041],[27.706,16.540,4.867],[27.541,16.540,5.215],[27.983,16.734,4.693],[27.734,16.540,5.215],[28.259,16.929,4.520],[28.450,14.205,2.086],[28.561,14.205,2.260],[28.672,14.399,2.434],[28.783,14.399,2.607],[28.894,14.205,2.781],[29.005,14.205,2.955],[29.198,14.205,2.955],[29.309,14.205,3.129],[29.503,14.205,3.129],[29.614,14.010,3.303],[29.807,14.010,3.303],[29.918,14.010,3.477],[30.029,13.815,3.650],[29.864,13.621,3.998],[29.781,13.426,4.172],[30.306,14.010,3.477],[30.720,14.205,2.607],[30.140,13.815,3.824],[29.698,13.232,4.346],[30.831,14.010,2.781],[30.582,14.010,3.303],[30.803,14.399,2.434],[29.809,13.037,4.520],[30.942,14.010,2.955],[30.859,14.010,3.129],[29.726,12.842,4.693],[31.053,14.010,3.129],[29.837,12.842,4.867],[31.163,14.010,3.303],[29.754,12.648,5.041],[29.865,12.453,5.215],[29.976,12.453,5.389],[25.740,14.788,3.303],[25.933,14.788,3.303],[26.210,14.983,3.129],[26.404,14.983,3.129],[26.597,15.178,3.129],[26.791,15.178,3.129],[26.985,15.178,3.129],[27.261,14.399,2.955],[27.261,14.594,2.955],[27.261,14.788,2.955],[27.261,14.983,2.955],[27.455,14.010,2.955],[27.455,14.205,2.955],[23.110,13.037,3.129],[23.221,13.037,3.303],[23.415,13.037,3.303],[23.526,13.037,3.477],[23.720,13.232,3.477],[23.720,13.426,3.477],[23.720,13.621,3.477],[23.968,13.815,2.955],[23.885,13.815,3.129],[23.803,13.815,3.303],[23.635,14.010,2.434],[23.829,14.010,2.434],[23.940,14.010,2.607],[24.051,14.010,2.781],[24.327,14.010,2.607],[23.525,14.010,2.260],[24.604,14.010,2.434],[24.880,14.205,2.260],[25.157,14.399,2.086],[25.268,14.205,2.260],[27.656,23.155,9.039],[26.821,15.567,4.693],[27.545,21.793,8.865],[27.462,21.793,9.039],[27.927,15.761,3.998],[28.120,15.567,3.998]]

