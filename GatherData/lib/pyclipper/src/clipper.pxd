cimport numpy as np
from cython.operator cimport dereference as deref
from libcpp cimport bool as cbool
from libcpp.vector cimport vector

cdef extern from "<string>" namespace "std":
    cdef cppclass string:
        string()
        string(string&)
        string(char*)

cdef extern from "clipper/clipper.h" namespace "clipper":
    cdef cppclass Coord_map
    cdef cppclass Coord_frac
    cdef cppclass Coord_orth

cdef extern from "clipper/clipper.h" namespace "clipper":
    cdef cppclass String:
        String()
        String(string)
        String(char*)
        char* data()

    cdef cppclass Vec3[T]:
        Vec3()
        Vec3 operator+(Vec3)
        Vec3 operator+(Coord_frac)
        Vec3 operator-(Vec3)
        float operator[](int)

    cdef cppclass Mat33:
        Mat33()

    cdef cppclass Grad_orth[T](Vec3):
        Grad_orth()

    cdef cppclass Grad_frac[T](Vec3):
        Grad_frac()
        Grad_orth grad_orth(Cell)
    cdef cppclass Grad_map[T](Vec3):
        Grad_map()
        Grad_frac grad_frac(Grid_sampling)
    cdef cppclass Cell:
        Cell()

    cdef cppclass RTop_orth:
        RTop_orth()

    cdef cppclass RTop_frac:
        Rtop_frac()
        Rtop_frac(Mat33, Vec3)
        RTop_orth rtop_orth(Cell)
        Vec3 trn()
        Mat33 rot()
        Vec3 _trn
        Mat33 _rot
        Vec3 operator*(Vec3)

    cdef cppclass Symop(RTop_frac):
        Symop()
        String format()

    cdef cppclass Spacegroup:
        Spacegroup()
        Symop symop(int)
        int num_symops()

    cdef cppclass Grid:
        Grid()
        int nu()
        int nv()
        int nw()

    cdef cppclass Grid_sampling(Grid):
        Grid_sampling()

    cdef cppclass Grid_range(Grid):
        Grid_range()
        cbool in_grid(Coord_grid)
    #cdef cppclass Interp_cubic

    cdef cppclass Xmap_base:
        cppclass Map_reference_base:
            Xmap_base base_xmap()
            int index()
            cbool last()
        cppclass Map_reference_index(Map_reference_base):
            Map_reference_index()
            Map_reference_index(Xmap_base)
            Map_reference_index(Xmap_base, Coord_grid)
            Coord_grid coord()
            Coord_orth coord_orth()
            Map_reference_index set_coord(Coord_grid)
            Map_reference_index next()
            int index_offset(int, int, int)
        cppclass Map_reference_coord(Map_reference_base):
            Map_reference_coord()
            Map_reference_coord(Xmap_base)
            Map_reference_coord(Xmap_base, Coord_grid)
            Coord_grid coord()
            Coord_orth coord_orth()
            int sym()
            Map_reference_coord set_coord(Coord_grid)
            Map_reference_coord next()
            Map_reference_coord next_u()
            Map_reference_coord next_v()
            Map_reference_coord next_w()
            Map_reference_coord prev_u()
            Map_reference_coord prev_v()
            Map_reference_coord prev_w()
        Map_reference_index first()
        int index_of(Coord_grid)
        Coord_grid coord_of(int)

    cdef cppclass Xmap[T](Xmap_base):
        Xmap()
        Cell cell()
        Grid_sampling grid_sampling()
        Grid_range grid_asu()
        Spacegroup spacegroup()
        T get_data(Coord_grid)
        T& operator[](Xmap_base.Map_reference_index)
        #T interp[](Coord_frac)

    cdef cppclass Coord_map:
        Coord_map()
        Coord_frac coord_frac(Grid_sampling)

    cdef cppclass Coord_grid:
        Coord_grid()
        Coord_grid(int, int, int)
        int u()
        int v()
        int w()
        Coord_grid unit(Grid_sampling)
        Coord_frac coord_frac(Grid_sampling)
        int operator==(Coord_grid)

    cdef cppclass Coord_frac(Vec3):
        Coord_frac()
        Coord_frac(float, float, float)
        Coord_map coord_map(Grid_sampling)
        Coord_grid coord_grid(Grid_sampling)
        Coord_orth coord_orth(Cell)
        Coord_frac lattice_copy_zero()
        Coord_frac operator-(Coord_frac)
        float lengthsq(Cell)
        Coord_frac lattice_copy_near()
        String format()
        float u()
        float v()
        float w()

    cdef cppclass Coord_orth(Vec3):
        Coord_orth()
        Coord_orth(np.float32_t, np.float32_t, np.float32_t)
        Coord_frac coord_frac(Cell)
        Coord_orth transform(RTop_orth)
        Coord_orth operator-(Coord_orth)
        String format()
        float x()
        float y()
        float z()

    cdef cppclass AtomShapeFn:
        AtomShapeFn()
        AtomShapeFn(Coord_orth, String, float, float)
        float rho(Coord_orth)

    cdef cppclass Atom:
        Atom()
        void set_element(String)
        void set_coord_orth(Coord_orth)
        void set_occupancy(float)
        void set_u_iso(float)

cdef extern from "clipper/contrib/skeleton.h" namespace "clipper":
    cdef cppclass Skeleton_fast[T1,T2]:
        cppclass Neighbours:
            Neighbours(Xmap_base&, float, float)

cdef extern from "clipper/clipper.h" namespace "clipper::Util":
    cdef float b2u(float)

cdef extern from "clipper/clipper.h" namespace "clipper::Interp_cubic":
    cdef void interp(Xmap[float], Coord_map&, float)
    cdef void interp_grad(Xmap[float], Coord_map&, float, Grad_map[float]&)
    #cdef void interp(Xmap[float], Coord_map&, np.float32_t)

cdef extern from "_clipper.h":
    cdef void find_clusters(Xmap[float]*, vector[Map_point_cluster]*, float, float, cbool)
    cdef void find_skeleton(Xmap[float]*, vector[Map_point_cluster]*, float, cbool)

    cdef cppclass Map_point_cluster:
        vector[Coord_grid] map_grid
        float score
        float max_ponint_score
        Coord_grid max_point
        Coord_grid max_point_box_compatible
        Xmap[float]* xmap

        Map_point_cluster(Xmap[float]*)
        float volume()
        float density()
        void add(Coord_grid)
        cbool is_compact()
        Coord_orth get_min_box_orth()
        Coord_orth get_max_box_orth()
        Coord_orth get_max_point_orth()
        Coord_orth get_max_point_box_orth()
        vector[Coord_orth] get_local_max_orth()
        vector[Coord_orth] get_skeleton()
        vector[Coord_orth] get_surface()

        Coord_frac get_min_box_frac()
        Coord_frac get_max_box_frac()
        Coord_frac get_max_point_frac()
        Coord_frac get_max_point_box_frac()

    cdef int exclude_non_standart_from_mmcif(char*, char*) except +
    cdef float import_xmap_from_mtz(Xmap[float]*, char*, char*, char*, char*, float, float) except +
    cdef void import_xmap_from_map(Xmap[float]*, char*) except +
    cdef void export_xmap(Xmap[float]*, char*) except +
    cdef Coord_orth transform_point(float, float, float, Symop, Cell, int, int, int)
    cdef Coord_orth transform_point_rel_to_000(float, float, float, Symop, Cell, int, int, int)
    cdef Coord_orth transform_point_inverse(float, float, float, Symop, Cell, int, int, int)
    cdef void get_map_mean_std_min_max(Xmap[float]*, float, float, float, float)
    cdef void get_map_statistics(Xmap[float]*, float, float, float, float, float, float, float)
    cdef float get_correlation(AtomShapeFn*, Coord_orth*, Xmap[float]*, float)
    cdef void cut_map_in_frac(Xmap[float]*, Xmap[float]*, float, float, float, float, float, float)
    cdef void set_value(Xmap[float]*, float)
    cdef void dist_map(Xmap[float]*, Xmap[float]*, vector[float]*, float)
