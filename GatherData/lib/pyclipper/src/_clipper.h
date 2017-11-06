#include <algorithm>
#include <clipper/clipper.h>
#include <vector>

int exclude_non_standart_from_mmcif(
                     char *c_mmcif_in_file_name,
                     char *c_mmcif_out_file_name
);

float import_xmap_from_mtz(
                     clipper::Xmap<float> *xmap,
                     char *c_mtz_file_name,
                     char *c_fo_col,
                     char *c_fphi_col,
                     char *c_weight_col,
                     float resolution_limit=0.0,
                     float target_grid_size=0.2
);

void import_xmap_from_map(
                     clipper::Xmap<float> *xmap,
                     char *c_map_file_name
);

void export_xmap(
    clipper::Xmap<float> *xmap,
    char *c_map_file_name
);


clipper::Coord_orth transform_point(float x, float y, float z,
                     clipper::Symop symop, clipper::Cell cell,
                     int x_shift, int y_shift, int z_shift);

clipper::Coord_orth transform_point_rel_to_000(float x, float y, float z,
                     clipper::Symop symop, clipper::Cell cell,
                     int x_shift, int y_shift, int z_shift
                     );

clipper::Coord_orth transform_point_inverse(float x, float y, float z,
                     clipper::Symop symop, clipper::Cell cell,
                     int x_shift, int y_shift, int z_shift);

void get_map_mean_std_min_max(clipper::Xmap<float> *xmap,
                             float &mean,
                             float &std,
                             float &min,
                             float &max
    );

void get_map_statistics(clipper::Xmap<float> *xmap,
                             float &p_mean,
                             float &p_square_mean,
                             float &p_cube_mean,
                             float &p_variance,
                             float &p_skewness,
                             float &p_min,
                             float &p_max
    );

float get_correlation(
        clipper::AtomShapeFn *ashape,
        clipper::Coord_orth *center,
        clipper::Xmap<float> *xmap,
        float radius
        );

void cut_map_in_frac(
        clipper::Xmap<float> *in_map,
        clipper::Xmap<float> *out_map,
        float min_frac_x,
        float min_frac_y,
        float min_frac_z,
        float max_frac_x,
        float max_frac_y,
        float max_frac_z
);


void dist_map(
        clipper::Xmap<float> *in_map,
        clipper::Xmap<float> *out_map,
        std::vector<float> *atoms,
        float radius = 2.5
);

void set_value(
        clipper::Xmap<float> *in_out_map,
        float value
);

void set_sqrt(
        clipper::Xmap<float> *in_out_map
);

class Map_point_cluster
{
    public:
        Map_point_cluster(clipper::Xmap<float> *xmap): xmap_ref(xmap) {
           score = 0.0;
           max_ponint_score = 0.0;
           max_point = clipper::Coord_grid();
           max_point_box_compatible = clipper::Coord_grid();
           map_grid = std::vector<clipper::Coord_grid>();
        };

        std::vector<clipper::Coord_grid> map_grid;


        float score;
        float max_ponint_score;
        clipper::Coord_grid max_point;
        clipper::Coord_grid max_point_box_compatible;
        clipper::Xmap<float> *xmap_ref;

        std::vector<clipper::Coord_orth> map_local_max_orth;
        std::vector<clipper::Coord_orth> skeleton;
        std::vector<clipper::Coord_orth> surface;

        bool operator==(const Map_point_cluster &mpc) const {
            return (mpc.map_grid == map_grid);
        }
        double volume() const;
        double density() const;
        clipper::Coord_orth grid_to_orth(const clipper::Coord_grid& c_g);
        void add(const clipper::Coord_grid& c_g);
        void add_local_max(const clipper::Coord_grid& c_g);
        void add_skeleton(const clipper::Coord_grid& c_g);
        void add_surface(const clipper::Coord_grid& c_g);

        clipper::Coord_orth get_min_box_orth();
        clipper::Coord_orth get_max_box_orth();
        clipper::Coord_orth get_max_point_orth();
        clipper::Coord_orth get_max_point_box_orth();
        std::vector<clipper::Coord_orth> get_local_max_orth();
        std::vector<clipper::Coord_orth> get_skeleton();
        std::vector<clipper::Coord_orth> get_surface();

        clipper::Coord_frac get_min_box_frac();
        clipper::Coord_frac get_max_box_frac();
        clipper::Coord_frac get_max_point_frac();
        clipper::Coord_frac get_max_point_box_frac();

        bool is_compact();
};

bool compare_clusters(const Map_point_cluster &a,
                      const Map_point_cluster &b);

void find_clusters(
        clipper::Xmap<float> *in_map,
        std::vector<Map_point_cluster> *cluster,
        float cut_off=6.0,
        float water_molecule_volume=11.0,
        bool verbose=false
);

void find_skeleton(
    clipper::Xmap<float> *in_map,
    std::vector<Map_point_cluster> *in_out_cluster,
    float cut_off,
    bool verbose=false
);