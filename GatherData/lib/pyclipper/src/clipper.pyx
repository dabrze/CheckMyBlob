# cython: profile=False
import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc

from clipper cimport *
from libcpp.vector cimport vector


cdef class ClipperMask(object):
    cdef Xmap[int] *cmap

    def __cinit__(self):
        self.cmap = new Xmap[int]()

    def __dealloc__(self):
        del self.cmap

    def get_values_p1(self):
        cdef Coord_grid cg
        cdef Coord_grid cga
        cdef Grid_sampling gr = self.cmap.grid_sampling()
        cdef int x = gr.nu(), y = gr.nv(), z = gr.nw()
        cdef Py_ssize_t ix, iy, iz = 0
        #cdef Xmap_base.Map_reference_index idx = self.cmap.first()
        cdef np.ndarray[np.int64_t, ndim=3] ar = np.empty((x,y,z), dtype=np.int64)
        ar[:] = np.NaN

        for ix in range(x):
            for iy in range(y):
                for iz in range(z):
                    cg = Coord_grid(ix, iy, iz)
                    ar[ix, iy, iz] = self.cmap.get_data(cg)
        return ar


cdef class ClipperMap(object):
    cdef Xmap[float] *cmap
    cdef float resolution

    def __cinit__(self, mtz=None, fo_col='FWT', fc_col='PHWT', weight_col='', map=None, resolution_limit=0.0, target_grid_size=0.2):
        self.cmap = new Xmap[float]()
        self.resolution = 0.0
        if mtz is not None:
            res = import_xmap_from_mtz(self.cmap, mtz, fo_col, fc_col, weight_col, resolution_limit, target_grid_size)
            self.resolution = res
        elif map is not None:
            import_xmap_from_map(self.cmap, map)

    def __dealloc__(self):
        del self.cmap

    def get_resolution(self):
        return self.resolution

    def save(self, map_filename):
        export_xmap(self.cmap, map_filename)

    def cut_map(self, min_frac_xyz, max_frac_xyz):
        cut_map = ClipperMap()
        cut_map.resolution = self.resolution

        cut_map_in_frac(
            self.cmap,
            cut_map.cmap,
            min_frac_xyz[0], min_frac_xyz[1], min_frac_xyz[2],
            max_frac_xyz[0], max_frac_xyz[1], max_frac_xyz[2]
        )
        return cut_map

    def get_map_index_at_points(self, np.ndarray[np.float32_t, ndim=2] points):
        cdef float x,y,z = 0
        cdef Coord_orth *co
        cdef Coord_frac fr
        cdef Coord_grid gr

        cdef Xmap_base.Map_reference_index idx = self.cmap.first()
        cdef np.ndarray[np.int_t, ndim=2] ar = np.empty((points.shape[0], points.shape[1]), dtype=np.int)

        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            z = points[i,2]
            co = new Coord_orth(x,y,z)
            fr = co.coord_frac(self.cmap.cell())
            gr = fr.coord_grid(self.cmap.grid_sampling())
            idx.set_coord(gr)
            gr = idx.coord()

            ar[i,0] = gr.u()
            ar[i,1] = gr.v()
            ar[i,2] = gr.w()
            del co
        return ar

    def get_values(self):
        #cdef Coord_grid *cg
        #cdef Coord_grid cga
        cdef Grid_range gr = self.cmap.grid_asu()
        cdef int x = gr.nu(), y = gr.nv(), z = gr.nw()
        #cdef Py_ssize_t ix, iy, iz = 0
        cdef Xmap_base.Map_reference_index idx = self.cmap.first()
        cdef np.ndarray[np.float32_t, ndim=3] ar = np.empty((x,y,z), dtype=np.float32)
        ar[:] = np.NaN

        while not idx.last():
            cg = idx.coord()
            ar[cg.u(), cg.v(), cg.w()] = deref(self.cmap)[idx]
            idx.next()

        #for ix in range(x):
        #    for iy in range(y):
        #        for iz in range(z):
        #            cg = new Coord_grid(ix, iy, iz)
        #            if self.cmap.index_of(deref(cg)) != -1:
        #            #cga = cg.coord_frac(gs).lattice_copy_zero().coord_grid(gs)
        #            #ar[cga.u(), cga.v(), cga.w()] = self.cmap.get_data(cga)
        #                ar[ix, iy, iz] = self.cmap.get_data(deref(cg))
        #
        #            del cg
        return ar

    def get_values_p1(self):
        cdef Coord_grid cg
        cdef Coord_grid cga
        cdef Grid_sampling gr = self.cmap.grid_sampling()
        cdef int x = gr.nu(), y = gr.nv(), z = gr.nw()
        cdef Py_ssize_t ix, iy, iz = 0
        # cdef Xmap_base.Map_refimport_xmap_from_mtzerence_index idx = self.cmap.first()
        cdef np.ndarray[np.float32_t, ndim=3] ar = np.empty((x,y,z), dtype=np.float32)
        ar[:] = np.NaN

        for ix in range(x):
            for iy in range(y):
                for iz in range(z):
                    cg = Coord_grid(ix, iy, iz)
                    ar[ix, iy, iz] = self.cmap.get_data(cg)
        return ar

    def get_map_data_for_points(self, type, np.ndarray[np.float32_t, ndim=2] points, np.ndarray[np.float32_t, ndim=2] result):
        cdef Py_ssize_t   i     = 0
        #cdef np.float32_t x,y,z = 0
        #cdef np.float32_t value = 0
        cdef float x,y,z = 0
        cdef float value = 0
        cdef Coord_orth *co
        cdef Coord_frac fr
        cdef Coord_map mp
        cdef Grid_sampling gs

        cdef Grad_frac[float] grad_frac
        cdef Grad_map[float] grad_map
        cdef Grad_orth[float] grad_orth
        cdef float gvalue

        cdef float mean
        cdef float variance
        cdef float min
        cdef float max

        raw, grad, corr = type
        type_positions = []
        for e in type:
            if e:
                type_positions.append(i)
                i+=1
            else:
                type_positions.append(-1)

        raw_pos = <Py_ssize_t> type_positions[0]
        grad_pos = <Py_ssize_t> type_positions[1]
        coor_pos = <Py_ssize_t> type_positions[2]

        if raw and not grad:
            raw_pos = type_positions[0]
            for i in range(points.shape[0]):
                x = points[i][0]
                y = points[i][1]
                z = points[i][2]

                co = new Coord_orth(x,y,z)
                fr = co.coord_frac(self.cmap.cell())
                mp = fr.coord_map(self.cmap.grid_sampling())
                interp(deref(self.cmap), mp, value)
                result[i, raw_pos] = value
                del co

        elif grad:
            raw_pos = type_positions[0]
            grad_pos = type_positions[1]
            for i in range(points.shape[0]):
                x = points[i][0]
                y = points[i][1]
                z = points[i][2]

                co = new Coord_orth(x,y,z)
                fr = co.coord_frac(self.cmap.cell())
                gs = self.cmap.grid_sampling()
                mp = fr.coord_map(gs)
                interp_grad(deref(self.cmap), mp, value, grad_map)
                grad_frac = grad_map.grad_frac(gs)
                grad_orth = grad_frac.grad_orth(self.cmap.cell())
                gvalue = sqrt(grad_orth[0]*grad_orth[0]+grad_orth[1]*grad_orth[1]+grad_orth[2]*grad_orth[2])
                result[i, raw_pos] = value
                result[i, grad_pos] = gvalue
                del co


    def get_correlation_for_atoms(self, np.ndarray[np.float32_t, ndim=2] atoms, np.ndarray elements,
                                  np.ndarray[np.float32_t, ndim=1] result):
        cdef float x,y,z,u,o = 0
        cdef float value = 0
        cdef Coord_orth *co
        cdef AtomShapeFn *ashape
        cdef float radius = 1.5
        for i in range(atoms.shape[0]):
            x = atoms[i][0]
            y = atoms[i][1]
            z = atoms[i][2]
            u = b2u(atoms[i][3])
            o = atoms[i][4]
            elem = new String(<char*>elements[i])
            co = new Coord_orth(x,y,z)
            ashape = new AtomShapeFn(deref(co), deref(elem), u, o);
            #atom.set_coord_orth(deref(co));
            #atom.set_element(deref(elem));
            #atom.set_u_iso(u);
            #atom.set_occupancy(o);
            value = get_correlation(ashape, co, self.cmap, radius)

            result[i] = value

            del ashape
            del elem
            del co

    def get_density_for_points2(self, np.ndarray[np.float32_t, ndim=2] points, np.ndarray[np.float32_t, ndim=1] result):
        cdef Py_ssize_t   i     = 0
        #cdef np.float32_t x,y,z = 0
        #cdef np.float32_t value = 0
        cdef float x,y,z = 0
        cdef float value = 0
        cdef Coord_orth *co
        cdef Coord_frac fr
        cdef Coord_map mp

        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            z = points[i,2]
            co = new Coord_orth(x,y,z)
            fr = co.coord_frac(self.cmap.cell())
            mp = fr.coord_map(self.cmap.grid_sampling())
            interp(deref(self.cmap), mp, value)
            result[i] = value
            #print co.format(),
            #print x, y, z
            del co
            #value = self.cmap.iterp[Interp_cubic](fr)

    def get_density_for_points(self, np.ndarray in_points):
        cdef Py_ssize_t i = 0
        cdef float x,y,z = 0
        cdef float value = 0
        cdef Coord_orth *co
        cdef Coord_frac fr
        cdef Coord_map mp

        cdef np.ndarray[np.float32_t, ndim=1] result = np.empty(in_points.shape[0], np.float32)

        for i in range(in_points.shape[0]):
            x = in_points[i,0]
            y = in_points[i,1]
            z = in_points[i,2]
            co = new Coord_orth(x,y,z)
            fr = co.coord_frac(self.cmap.cell())
            mp = fr.coord_map(self.cmap.grid_sampling())
            interp(deref(self.cmap), mp, value)
            result[i] = value
            del co
        return result

    def get_density_for_point(self, float x, float y, float z):
        cdef float value = 0.0
        orth = new Coord_orth(x, y, z)
        frac = orth.coord_frac(self.cmap.cell())
        map_ppoint = frac.coord_map(self.cmap.grid_sampling())
        interp(deref(self.cmap), map_ppoint, value)
        del orth
        return value

    def get_points_in_asu(self, np.ndarray[np.float32_t, ndim=2] points, np.ndarray[np.float32_t, ndim=2] result):
        cdef Py_ssize_t   i     = 0
        #cdef np.float32_t x,y,z = 0
        #cdef np.float32_t value = 0
        cdef float x,y,z = 0
        cdef float value = 0
        cdef Coord_orth *co
        cdef Coord_orth co2
        cdef Coord_frac fr
        cdef Coord_map mp

        for i in range(points.shape[0]):
            x, y, z = points[i]
            co = new Coord_orth(x, y, z)
            fr = co.coord_frac(self.cmap.cell())
            mp = fr.coord_map(self.cmap.grid_sampling())
            fr = mp.coord_frac(self.cmap.grid_sampling())
            co2 = fr.coord_orth(self.cmap.cell())
            result[i, 0] = co2.x()
            result[i, 1] = co2.y()
            result[i, 2] = co2.z()
            del co

    def get_shift_to_origin(self, np.ndarray[np.float_t, ndim=1] point):
        """ Get vector which translates point near the cell origin """
        # Get fractional coords
        cdef Coord_frac cf
        cdef Coord_frac ocf
        cdef Coord_frac shiftf
        cdef Coord_orth* co
        cdef Coord_orth shift

        co = new Coord_orth(point[0], point[1], point[2])
        cf = co.coord_frac(self.cmap.cell())
        ocf = cf.lattice_copy_zero()
        shiftf = ocf - cf
        shift = shiftf.coord_orth(self.cmap.cell())

        result = np.array((shift.x(), shift.y(), shift.z()))
        del co

        return result

    def transform_all(self, np.ndarray[np.float_t, ndim=2] points):
        cdef Py_ssize_t   j, k = 0, t = 0
        cdef int i = 0
        cdef Py_ssize_t l = points.shape[0]
        cdef float x, y, z = 0
        cdef Spacegroup sg      = self.cmap.spacegroup()
        cdef Symop so
        cdef Coord_orth co_tr
        cdef int x_shift, y_shift, z_shift = -1
        cdef int num_symops     = sg.num_symops()

        unit_cell_shifts = (0, -1, 1)
        cdef np.ndarray[np.float32_t, ndim = 2] transformed_points = np.zeros((l*(num_symops*27), 3), dtype=np.float32)
        for i in range(num_symops):
            so = sg.symop(i)
            for x_shift in unit_cell_shifts:
                for y_shift in unit_cell_shifts:
                    for z_shift in unit_cell_shifts:
                        t += 1
                        for j in range(l):
                            x, y, z = points[j]
                            co_tr = transform_point(x, y, z, so, self.cmap.cell(), x_shift, y_shift, z_shift)
                            transformed_points[k, 0] = co_tr.x()
                            transformed_points[k, 1] = co_tr.y()
                            transformed_points[k, 2] = co_tr.z()
                            k += 1
        return transformed_points

    def transform_all_rel_to_000(self, np.ndarray[np.float_t, ndim=2] points):
        cdef Py_ssize_t   j, k = 0
        cdef int i = 0
        cdef Py_ssize_t l = points.shape[0]
        cdef float x, y, z = 0
        cdef Spacegroup sg = self.cmap.spacegroup()
        cdef Symop so
        cdef Coord_orth co_tr
        cdef int x_shift, y_shift, z_shift = -1
        cdef int num_symops = sg.num_symops()

        unit_cell_shifts = (0, -1, 1)
        cdef np.ndarray[np.float32_t, ndim = 2] transformed_points = np.zeros((l*(num_symops*27), 3), dtype=np.float32)

        for i in range(num_symops):
            so = sg.symop(i)
            for x_shift in unit_cell_shifts:
                for y_shift in unit_cell_shifts:
                    for z_shift in unit_cell_shifts:
                        for j in range(l):
                            x, y, z = points[j]
                            co_tr = transform_point_rel_to_000(x, y, z, so, self.cmap.cell(), x_shift, y_shift, z_shift)
                            transformed_points[k, 0] = co_tr.x()
                            transformed_points[k, 1] = co_tr.y()
                            transformed_points[k, 2] = co_tr.z()
                            k += 1
        return transformed_points

    def transform_one(self, np.ndarray[np.float_t, ndim=2] points, int num, int inverse=0):
        cdef Py_ssize_t   j, k    = 0
        cdef int i = 0
        cdef Py_ssize_t l = points.shape[0]
        cdef float x, y, z = 0
        cdef Spacegroup sg      = self.cmap.spacegroup()
        cdef Symop so
        cdef Coord_orth co_tr
        cdef int x_shift, y_shift, z_shift = -1
        cdef int num_symops     = sg.num_symops()
        cdef int count          = 0

        cdef np.ndarray[np.float32_t, ndim=2] transformed_points = np.array(points, dtype=np.float32, copy=True)

        if inverse == 0:
            transform_point_func = transform_point
        else:
            transform_point_func = transform_point_inverse
        unit_cell_shifts = (0, -1, 1)

        for i in range(num_symops):
            so = sg.symop(i)
            for x_shift in unit_cell_shifts:
                for y_shift in unit_cell_shifts:
                    for z_shift in unit_cell_shifts:
                        if count == num:
                            for j in range(l):
                                x, y, z = points[j]
                                co_tr = transform_point_func(x, y, z, so, self.cmap.cell(), x_shift, y_shift, z_shift)
                                #print k, j, (i*l), ((z_shift+1)*l), ((y_shift+1)*3*l), ((x_shift+1)*9*l)
                                transformed_points[k, 0] = co_tr.x()
                                transformed_points[k, 1] = co_tr.y()
                                transformed_points[k, 2] = co_tr.z()
                                k += 1
                            return transformed_points
                        count += 1
        return transformed_points

    def get_mean_std_min_max(self):
        cdef float mean = 0
        cdef float std = 0
        cdef float min = 0
        cdef float max = 0

        get_map_mean_std_min_max(self.cmap, mean, std, min, max)

        return mean, std, min, max

    def get_statistics(self):
        cdef float mean = 0.0
        cdef float square_mean = 0.0
        cdef float cube_mean = 0.0
        cdef float variance = 0.0
        cdef float skewness = 0.0
        cdef float min = 0.0
        cdef float max = 0.0

        get_map_statistics(self.cmap, mean, square_mean, cube_mean, variance, skewness, min, max)
        return mean, square_mean, cube_mean, variance, skewness, min, max

    def get_atoms_within_box(self, np.ndarray[np.float_t, ndim=2] points, float min_x, float min_y, float min_z, float max_x, float max_y, float max_z):
        cdef int i = 0
        cdef np.ndarray[np.float32_t, ndim = 2] transformed = self.transform_all(points)
        cdef Py_ssize_t transformed_shape = transformed.shape[0]

        cdef np.ndarray[np.float32_t, ndim = 2] points_inside = np.empty((0, 3), np.float32)
        for i in range(transformed_shape):
            point = transformed[i]
            if ((point[0] > min_x) and (point[1] > min_y) and (point[2] > min_z) and
               (point[0] < max_x) and (point[1] < max_y) and (point[2] < max_z)):
                np.append(points_inside, [point])
        return points_inside

    def get_dist_map(self, np.ndarray[np.float_t, ndim=2] points, float max_radii):
        d_map = ClipperMap()
        d_map.resolution = self.resolution

        cdef vector[float]* atoms = new vector[float]()
        cdef int i
        cdef Py_ssize_t l = points.shape[0]

        for j in range(l):
            x, y, z = points[j]
            atoms.push_back(x)
            atoms.push_back(y)
            atoms.push_back(z)

        dist_map(self.cmap, d_map.cmap, atoms, max_radii)
        del atoms
        return d_map

    cdef _get_list_of_points(self, vector[Coord_orth] vec):
        vec_list = list()
        vec_it = vec.begin()
        while vec_it != vec.end():
            vec_list.append(tuple([deref(vec_it).x(), deref(vec_it).y(), deref(vec_it).z()]))
            inc(vec_it)
        return vec_list

    def get_mask_map(self, float cut_off, float water_molecule_volume, cbool custom_skeleton=False, cbool verbose=False):
        cluster_list = []
        cdef vector[Map_point_cluster]* clusters = new vector[Map_point_cluster]()
        find_clusters(self.cmap, clusters, cut_off, water_molecule_volume, verbose)
        if custom_skeleton is True:
            find_skeleton(self.cmap, clusters, cut_off, verbose)

        cdef vector[Map_point_cluster].iterator it = clusters.begin()
        cdef vector[Coord_orth].iterator local_max_it

        while it != clusters.end():
            min_box = deref(it).get_min_box_frac()
            max_box = deref(it).get_max_box_frac()
            max_point = deref(it).get_max_point_frac()
            min_box_o = deref(it).get_min_box_orth()
            max_box_o = deref(it).get_max_box_orth()
            max_point_o = deref(it).get_max_point_orth()
            max_point_box_o = deref(it).get_max_point_box_orth()

            local_max_o_list = self._get_list_of_points(deref(it).get_local_max_orth())
            skeleton_list = self._get_list_of_points(deref(it).get_skeleton())
            surface_list = self._get_list_of_points(deref(it).get_surface())

            cluster_list.append(MapCluster(
                deref(it).volume(),
                deref(it).density(),
                min_box.u(), min_box.v(), min_box.w(),
                max_box.u(), max_box.v(), max_box.w(),
                max_point.u(), max_point.v(), max_point.w(),
                min_box_o.x(), min_box_o.y(), min_box_o.z(),
                max_box_o.x(), max_box_o.y(), max_box_o.z(),
                max_point_o.x(), max_point_o.y(), max_point_o.z(),
                max_point_box_o.x(), max_point_box_o.y(), max_point_box_o.z(),
                local_max_o_list,
                skeleton_list,
                surface_list,
            ))
            inc(it)
        del clusters

        return cluster_list

    def get_closest_idx(self, np.ndarray[np.float_t, ndim=2] transformed_position, np.ndarray[np.int_t, ndim=1] atoms_idx, np.ndarray[np.float_t, ndim=1] max_point_in_orth):
        cdef Py_ssize_t l = transformed_position.shape[0]
        cdef int atom_idx
        cdef int i_idx
        cdef np.ndarray[np.float_t, ndim=1] diff
        cdef float dist
        cdef float min_dist
        cdef int min_idx

        min_idx = -1
        min_dist = 99999999.0
        for i_idx in range(l):
            atom_idx = atoms_idx[i_idx]
            diff = transformed_position[i_idx]-max_point_in_orth
            dist = np.dot(diff, diff)
            if dist < min_dist:
                min_idx = atom_idx
                min_dist = dist
        if min_idx == -1:
            return None, None
        return min_idx, np.sqrt(min_dist)


cdef class CifCut:

    @staticmethod
    def exclude_res_from_mmcif(in_mmcif, out_mmcif):
        return exclude_non_standart_from_mmcif(in_mmcif, out_mmcif)


cdef class MapCluster:

    cdef public float volume
    cdef public float density
    cdef public object min_box_f
    cdef public object max_box_f
    cdef public object max_point_f
    cdef public object min_box_o
    cdef public object max_box_o
    cdef public object max_point_o
    cdef public object max_point_box_o
    cdef public list map_local_max_o
    cdef public list skeleton
    cdef public list surface

    def __init__(self, float volume, float density,
                 float min_box_x_f, float min_box_y_f, float min_box_z_f,
                 float max_box_x_f, float max_box_y_f, float max_box_z_f,
                 float max_point_x_f, float max_point_y_f, float max_point_z_f,
                 float min_box_x_o, float min_box_y_o, float min_box_z_o,
                 float max_box_x_o, float max_box_y_o, float max_box_z_o,
                 float max_point_x_o, float max_point_y_o, float max_point_z_o,
                 float max_point_box_x_o, float max_point_box_y_o, float max_point_box_z_o,
                 list map_local_max_o, list skeleton, list surface):
        self.volume = volume
        self.density = density
        self.min_box_f = (min_box_x_f, min_box_y_f, min_box_z_f)
        self.max_box_f = (max_box_x_f, max_box_y_f, max_box_z_f)
        self.max_point_f = (max_point_x_f, max_point_y_f, max_point_z_f)
        self.min_box_o = (min_box_x_o, min_box_y_o, min_box_z_o)
        self.max_box_o = (max_box_x_o, max_box_y_o, max_box_z_o)
        self.max_point_o = (max_point_x_o, max_point_y_o, max_point_z_o)
        self.max_point_box_o = (max_point_box_x_o, max_point_box_y_o, max_point_box_z_o)
        self.map_local_max_o = map_local_max_o
        self.skeleton = skeleton
        self.surface = surface


cdef class ClipperAtomShape:

    cdef AtomShapeFn *thisptr

    def __cinit__(self, float x, float  y, float z, string elem, float uiso, float occu):
        co = new Coord_orth(x, y, z)
        element = new String(elem)
        self.thisptr = new AtomShapeFn(deref(co), deref(element), uiso, occu)
        del co
        del element

    def __dealloc__(self):
        del self.thisptr

    def rho(self, float x, float  y, float z):
        co = new Coord_orth(x, y, z)
        cdef float rho = self.thisptr.rho(deref(co))
        del co
        return rho


cdef extern from "math.h":
    np.float_t sqrt(np.float_t n)
