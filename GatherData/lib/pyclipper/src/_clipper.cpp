#include <clipper/clipper.h>
#include <clipper/clipper-ccp4.h>
#include <clipper/contrib/skeleton.h>
#include <clipper/mmdb/clipper_mmdb.h>
#include <numeric>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string.h>

#include "_clipper.h"

typedef unsigned char pixel_type;


int exclude_non_standart_from_mmcif(
                     char *c_mmcif_in_file_name,
                     char *c_mmcif_out_file_name
){
  clipper::mmdb::CMMDBManager manager;
  int                   im, ic, ir, nModels, nChains, nResidues;
  clipper::mmdb::PPCModel       model;
  clipper::mmdb::PPCChain       chain;
  clipper::mmdb::PPCResidue     res;

  int err = manager.ReadCoorFile (c_mmcif_in_file_name);
  if (err) {
    return err;
  }

  //  get table of models
  manager.GetModelTable(model, nModels);

  //  loop over all models
  for (im=0; im < nModels; im++)
  {
    if (model[im])
    {
      //  get chain table of im-th model
      model[im]->GetChainTable(chain, nChains);
      //  loop over all chains:
      for (ic=0; ic < nChains; ic++)
      {
        if (chain[ic])
        {
          // get residue table for current chain:
          chain[ic]->GetResidueTable(res, nResidues);
          // loop over all residues in current chain:
          for (ir=0; ir < nResidues; ir++)
          {
            // delete all oxygens in the residue
            if (res[ir])
            {
              std::string res_name = res[ir]->GetResName();
              if (not (
                  res_name.compare("ALA") == 0 ||
                  res_name.compare("ARG") == 0 ||
                  res_name.compare("ASN") == 0 ||
                  res_name.compare("ASP") == 0 ||
                  res_name.compare("CSH") == 0 ||
                  res_name.compare("CYS") == 0 ||
                  res_name.compare("GLN") == 0 ||
                  res_name.compare("GLU") == 0 ||
                  res_name.compare("GLY") == 0 ||
                  res_name.compare("HIS") == 0 ||
                  res_name.compare("ILE") == 0 ||
                  res_name.compare("LEU") == 0 ||
                  res_name.compare("LYS") == 0 ||
                  res_name.compare("MET") == 0 ||
                  res_name.compare("MSE") == 0 ||
                  res_name.compare("ORN") == 0 ||
                  res_name.compare("PHE") == 0 ||
                  res_name.compare("PRO") == 0 ||
                  res_name.compare("SER") == 0 ||
                  res_name.compare("THR") == 0 ||
                  res_name.compare("TRP") == 0 ||
                  res_name.compare("TYR") == 0 ||
                  res_name.compare("VAL") == 0 ||
                  res_name.compare("DA") == 0 ||
                  res_name.compare("DG") == 0 ||
                  res_name.compare("DT") == 0 ||
                  res_name.compare("DC") == 0 ||
                  res_name.compare("A") == 0 ||
                  res_name.compare("G") == 0 ||
                  res_name.compare("T") == 0 ||
                  res_name.compare("C") == 0 ||
                  res_name.compare("U") == 0
              ))
              {
                //res[ir]->DeleteAllAtoms();
                chain[ic]->DeleteResidue(ir);
              }
            }
          }
        }
      }
    }
  }

  //  update internal references;
  manager.FinishStructEdit();

  err = manager.WriteCIFASCII(c_mmcif_out_file_name);
  if (err) {
    return err;
  }
  return 0;
}

float import_xmap_from_mtz(
                     clipper::Xmap<float> *xmap,
                     char *c_mtz_file_name,
                     char *c_fo_col,
                     char *c_fphi_col,
                     char *c_weight_col,
                     float resolution_limit,
                     float target_grid_size
)
{
  std::string mtz_file_name(c_mtz_file_name);
  std::string fo_col(c_fo_col);
  std::string fphi_col(c_fphi_col);
  std::string weight_col(c_weight_col);
  bool no_weight = weight_col.empty();

  clipper::HKL_info hkls;

  clipper::CCP4MTZfile mtzfile;


  if (!no_weight) {
      fphi_col = "/*/*/["+fphi_col+" "+weight_col+"]";
  } else {
      fphi_col = "/*/*/["+fo_col+" "+fphi_col+"]";
  }
  fo_col = "/*/*/["+fo_col+" SIGF]";
  //fo_col = "/*/*/["+fo_col+" "+fo_col+"]";

  std::cout << fo_col << fphi_col << no_weight << "\n";
  //mtzfile.set_column_label_mode( clipper::CCP4MTZfile::Legacy );
  mtzfile.open_read( mtz_file_name );
  mtzfile.import_hkl_info( hkls );
  clipper::HKL_data< clipper::datatypes::F_sigF<float>  > fo(hkls);
  clipper::HKL_data< clipper::datatypes::Phi_fom<float> > phi_fom(hkls);
  clipper::HKL_data< clipper::datatypes::F_phi<float>   > fphi(hkls);

  if (!no_weight) {
      std::cout << "Using weight" << "\n";
      mtzfile.import_hkl_data( fo, fo_col );
      mtzfile.import_hkl_data( phi_fom, fphi_col);
      mtzfile.close_read();
      fphi.compute(fo, phi_fom,
               clipper::datatypes::Compute_fphi_from_fsigf_phifom<float>());
  } else {
      std::cout << "No weight" << "\n";
      mtzfile.import_hkl_data( fphi, fphi_col );
      mtzfile.close_read();
  }
  //mtzfile.close_read();

  clipper::Resolution resolution_file;
  resolution_file = clipper::Resolution(1.0/sqrt(fphi.invresolsq_range().max()));

  clipper::Resolution resolution;
  //resolution = clipper::Resolution(std::max(1.0/sqrt(fphi.invresolsq_range().max()), double(resolution_limit)));
  resolution = clipper::Resolution(std::max(1.0/sqrt(fphi.invresolsq_range().max()), double(resolution_limit)));

  if (resolution_file.limit() < resolution.limit())
  {
    clipper::HKL_info new_hkls(hkls.spacegroup(), hkls.cell(), resolution, true);
    clipper::HKL_data<clipper::datatypes::F_phi<float> > new_fphi(hkls);
    clipper::HKL_info::HKL_reference_index ih;
    clipper::HKL_info::HKL_reference_coord ik;
    for ( ih = new_hkls.first(); !ih.last(); ih.next() ) {
      new_fphi[ih] = fphi[ih.hkl()];
    }
    //float invopt = 1/std::sqrt(resolution.limit());
    //for ( ih = hkls.first(); !ih.last(); ih.next() ) {
    //if (ih.invresolsq() > invopt)
      //    fphi[ih].set_null();
    //}
    fphi = new_fphi;
    hkls = new_hkls;
  }

  // rate 1.5 is default for 2A it means grid size is approximately 0.666(6)A
  double rate = std::max(0.5*resolution.limit() / target_grid_size, 1.5);
  xmap->init( hkls.spacegroup(), hkls.cell(),
              clipper::Grid_sampling( hkls.spacegroup(), hkls.cell(),
              resolution, rate));
  //std::cout << "FFT" << "\n";
  xmap->fft_from( fphi );
  return resolution_file.limit();
}


void import_xmap_from_map(
                     clipper::Xmap<float> *xmap,
                     char *c_map_file_name
)
{
    clipper::CCP4MAPfile file;
    file.open_read(c_map_file_name);
    file.import_xmap(*xmap);
    file.close_read();
}


void export_xmap(clipper::Xmap<float> *xmap, char *c_map_file_name)
{
    clipper::CCP4MAPfile mapfile;
    mapfile.open_write(c_map_file_name);
    mapfile.export_xmap(*xmap);
    mapfile.close_write();
}


void get_map_mean_std_min_max(clipper::Xmap<float> *xmap,
                              float &p_mean,
                              float &p_std,
                              float &p_min,
                              float &p_max
)
{
  // Calculate map statistics

  float min = 1e10, max = -1e10, sum = 0.0, sum_sq = 0;
  float v;
  float n_point = 0.0;
  float mean, variance;

   clipper::Xmap_base::Map_reference_index ix;
   for (ix=xmap->first(); !ix.last(); ix.next()) {

      n_point += 1.0;
      v = (*xmap)[ix];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      sum_sq += v*v;

   }
   mean     = float( sum/n_point );
   variance = float( (n_point*sum_sq - sum*sum) / (n_point*n_point) );
   p_mean = mean;
   p_std = sqrt(variance);
   p_min = min;
   p_max = max;
};


void get_map_statistics(clipper::Xmap<float> *xmap,
                        float &p_mean,
                        float &p_square_mean,
                        float &p_cube_mean,
                        float &p_variance,
                        float &p_skewness,
                        float &p_min,
                        float &p_max
    )
{
    const int num_symops = xmap->spacegroup().num_symops();
    int copies = 0;
    long long n_count = 0;

    double mean = 0.0; // usually 0.0 for the whole map
    double denisty_square_mean = 0.0;
    double density_cube_mean = 0.0;

    double minimum =  1.0e20;
    double maximum = -1.0e20;

    double density = 0.0;

    clipper::Xmap_base::Map_reference_index ix;
    for ( ix = xmap->first(); !ix.last(); ix.next() )
    {
        copies = num_symops / xmap->multiplicity( ix.coord() );
        n_count += copies;

        density = (*xmap)[ix];

        double copies_density = copies*density;
        double copies_density_square = copies_density*density;

        mean += copies_density;
        denisty_square_mean += copies_density_square;
        density_cube_mean   += copies_density_square*density;

        if (density < minimum)
        {
            minimum = density;
        }
        if (density > maximum)
        {
            maximum = density;
        }
    }

    /*
    double central_moment = 0.0;
    double variance = 0.0;
    double skewness = 0.0;

    clipper::Xmap<float>::Map_reference_index ix( xmap );
    for ( ix = xmap->first(); !ix.last(); ix.next() )
    {
        copies = num_symops / xmap->multiplicity( ix.coord() );
        n_count += copies;

        density = (*xmap)[ix];

        double density_minus_mean = (density-mean);
        double copies_density_minus_mean = copies*density_minus_mean;
        double copies_density_square_minus_mean = copies_density_minus_mean*density_minus_mean;
        central_moment += copies_density_minus_mean;
        variance       += copies_density_square_minus_mean;
        skewness       += copies_density_square_minus_mean*density_minus_mean;
    }
    p_variance       = variance/double(n_count);
    p_skewness       = skewness/double(n_count);
    */

    p_mean           = mean/double(n_count);
    p_square_mean    = denisty_square_mean/double(n_count);
    p_cube_mean      = density_cube_mean/double(n_count);
    p_variance       = p_square_mean - p_mean*p_mean; //variance/double(n_count);
    p_skewness       = p_cube_mean - 3.0*p_mean*p_square_mean + 2.0*p_mean*p_mean*p_mean; //skewness/double(n_count);
    p_min = minimum;
    p_max = maximum;
};


clipper::Coord_orth transform_point(float x, float y, float z,
                     clipper::Symop symop, clipper::Cell cell,
                     int x_shift, int y_shift, int z_shift)
{
    clipper::Coord_orth co(x, y, z);
    clipper::Coord_frac cf = co.coord_frac(cell);
    clipper::Coord_frac cell_shift(x_shift, y_shift, z_shift);
    clipper::RTop_frac rtf(symop.rot(), symop.trn());

    clipper::Coord_frac ctr;
    ctr = cf.transform(rtf);
    // integer part + fractional part + shift
    ctr = cf - cf.lattice_copy_unit() + ctr.lattice_copy_unit() + cell_shift;
    return(ctr.coord_orth(cell));
};

clipper::Coord_orth transform_point_rel_to_000(float x, float y, float z,
                     clipper::Symop symop, clipper::Cell cell,
                     int x_shift, int y_shift, int z_shift)
{
    clipper::Coord_orth co(x, y, z);
    clipper::Coord_frac cf = co.coord_frac(cell);
    clipper::Coord_frac cell_shift(x_shift, y_shift, z_shift);
    clipper::RTop_frac rtf(symop.rot(), symop.trn());
    clipper::Coord_frac ctr;
    ctr = cf.transform(rtf);

    // integer part + fractional part + shift
    ctr = ctr.lattice_copy_unit() + cell_shift;
    return(ctr.coord_orth(cell));
};

clipper::Coord_orth transform_point_inverse(float x, float y, float z,
                     clipper::Symop symop, clipper::Cell cell,
                     int x_shift, int y_shift, int z_shift)
{
    clipper::Coord_orth co(x, y, z);
    clipper::Coord_frac cf = co.coord_frac(cell);
    clipper::Coord_frac cell_shift(x_shift, y_shift, z_shift);
    clipper::RTop_frac rtf(symop.rot(), symop.trn());

    clipper::Coord_frac ctr;
    ctr = cf.transform(rtf.inverse());
    // integer part + fractional part - shift
    ctr = cf - cf.lattice_copy_unit() + ctr.lattice_copy_unit() - cell_shift;
    return(ctr.coord_orth(cell));
};


float get_correlation(clipper::AtomShapeFn *ashape, clipper::Coord_orth *center, clipper::Xmap<float> *map, float radius)
{
    //clipper::AtomShapeFn ashape(atom.coord_orth(), atom.element(), atom.u_iso(), atom.occupancy());
    clipper::Coord_orth o0(center->x() - radius, center->y() - radius,  center->z() - radius);
    clipper::Coord_frac f0(o0.coord_frac(map->cell()));
    clipper::Coord_grid g0(f0.coord_grid(map->grid_sampling()));
    clipper::Coord_orth o1(center->x() + radius, center->y() + radius,  center->z() + radius);
    clipper::Coord_frac f1(o1.coord_frac(map->cell()));
    clipper::Coord_grid g1(f1.coord_grid(map->grid_sampling()));

    std::vector<float> map_rhos, atom_rhos;
    float map_sum = 0, atom_sum = 0;
    float t1 = 0, t2 = 0, t3 = 0;
    float dm = 0, da = 0;

    clipper::Xmap_base::Map_reference_coord i0, iu, iv, iw;
    i0 = clipper::Xmap_base::Map_reference_coord( *map, g0 );
    for ( iu = i0; iu.coord().u() <= g1.u(); iu.next_u() )
        for ( iv = iu; iv.coord().v() <= g1.v(); iv.next_v() )
            for ( iw = iv; iw.coord().w() <= g1.w(); iw.next_w() ) {
                std::cout.flush();
                map_rhos.push_back(map->get_data(iw.coord()));
                atom_rhos.push_back(ashape->rho(iw.coord_orth()));
    }
    std::accumulate(map_rhos.begin(), map_rhos.end(), map_sum);
    std::accumulate(atom_rhos.begin(), atom_rhos.end(), atom_sum);

    std::vector<float>::iterator it1;
    std::vector<float>::iterator it2;
    for (
      it1 = map_rhos.begin(),
      it2 = atom_rhos.begin();
      it1 != map_rhos.end(), it2 != atom_rhos.end();
      ++it1, ++it2) {
        dm = *it1 - map_sum;
        da = *it2 - atom_sum;
        t1 += dm*da;
        t2 += dm*dm;
        t3 += da*da;
    }

    return(t1/(sqrt(t2)*sqrt(t3)));
};

void cut_map_in_frac(
        clipper::Xmap<float> *in_map,
        clipper::Xmap<float> *out_map,
        float min_frac_x,
        float min_frac_y,
        float min_frac_z,
        float max_frac_x,
        float max_frac_y,
        float max_frac_z
)
{
    clipper::Cell cut_cell( in_map->cell() );
    clipper::Spacegroup cut_spacegroup( clipper::Spacegroup::P1 );
    clipper::Grid_sampling cut_grid_sampling( in_map->grid_sampling() );
    out_map->init( cut_spacegroup, cut_cell, cut_grid_sampling);

    clipper::Coord_frac frac_min(min_frac_x, min_frac_y, min_frac_z);
    clipper::Coord_grid grid_min(frac_min.coord_grid(in_map->grid_sampling()));
    clipper::Coord_frac frac_max(max_frac_x, max_frac_y, max_frac_z);
    clipper::Coord_grid grid_max(frac_max.coord_grid(in_map->grid_sampling()));

    //clipper::Grid_range grid_range(in_map->grid_sampling(), frac_min, frac_max);

    // fill the cut map
    clipper::Xmap_base::Map_reference_coord i0, iu, iv, iw;
    i0 = clipper::Xmap_base::Map_reference_coord(*out_map, grid_min);
    for ( iu = i0; iu.coord().u() <= grid_max.u(); iu.next_u() )
      for ( iv = iu; iv.coord().v() <= grid_max.v(); iv.next_v() )
        for ( iw = iv; iw.coord().w() <= grid_max.w(); iw.next_w() )
        {
            clipper::Coord_orth cut_coord_otrh = iw.coord_orth();
            //(*out_map)[iw] = (*in_map)[iw] //interp<clipper::Interp_cubic>(cut_coord_frac);
            (*out_map)[iw] = in_map->interp<clipper::Interp_cubic>(in_map->coord_map(cut_coord_otrh));
        }
}

void set_value(
        clipper::Xmap<float> *in_out_map,
        float value
)
{
    clipper::Xmap<float>::Map_reference_index inx;
    for (inx = in_out_map->first(); !inx.last(); inx.next()) {
        (*in_out_map)[inx] = value;
    }
}

void set_sqrt(
        clipper::Xmap<float> *in_out_map
)
{
    clipper::Xmap<float>::Map_reference_index inx;
    for (inx = in_out_map->first(); !inx.last(); inx.next()) {
        (*in_out_map)[inx] = sqrt((*in_out_map)[inx]);
    }
}

void dist_map(
        clipper::Xmap<float> *in_map,
        clipper::Xmap<float> *out_map,
        std::vector<float> *atoms,
        float radius
)
{
    clipper::Cell cell( in_map->cell() );
    clipper::Spacegroup spacegroup( clipper::Spacegroup::P1 );
    clipper::Spacegroup in_map_spacegroup = in_map->spacegroup();
    //clipper::Resolution resolution(2.0);
    //clipper::Grid_sampling grid_sampling(in_map->spacegroup(), in_map->cell(), resolution);
    clipper::Grid_sampling grid_sampling( in_map->grid_sampling() );
    out_map->init(spacegroup, cell, grid_sampling);

    //fill with default
    float dist_max = radius*radius*radius*radius;
    set_value(out_map, dist_max);


    clipper::Coord_map one100 = clipper::Coord_map(1,0,0);
    clipper::Coord_orth orth100 = out_map->coord_orth(one100);
    float len = sqrt(orth100.lengthsq());
    clipper::Coord_orth orthR00 = (radius/len)*orth100;
    clipper::Coord_map oneR00 = out_map->coord_map(orthR00);
    clipper::Coord_grid gridR00 = oneR00.ceil();

    clipper::Coord_map one010 = clipper::Coord_map(0,1,0);
    clipper::Coord_orth orth010 = out_map->coord_orth(one010);
    len = sqrt(orth010.lengthsq());
    clipper::Coord_orth orth0R0 = (radius/len)*orth010;
    clipper::Coord_map one0R0 = out_map->coord_map(orth0R0);
    clipper::Coord_grid grid0R0 = one0R0.ceil();

    clipper::Coord_map one001 = clipper::Coord_map(0,0,1);
    clipper::Coord_orth orth001 = out_map->coord_orth(one001);
    len = sqrt(orth001.lengthsq());
    clipper::Coord_orth orth00R = (radius/len)*orth001;
    clipper::Coord_map one00R = out_map->coord_map(orth00R);
    clipper::Coord_grid grid00R = one00R.ceil();

    int u = std::max(std::max(gridR00.u(), grid0R0.u()), grid00R.u())+1;
    int v = std::max(std::max(gridR00.v(), grid0R0.v()), grid00R.v())+1;
    int w = std::max(std::max(gridR00.w(), grid0R0.w()), grid00R.w())+1;
    clipper::Coord_grid box_max = clipper::Coord_grid(u, v, w);
    clipper::Coord_grid box_min = clipper::Coord_grid(-u, -v, -w);

    //clipper::Grid_range atom_box( grid_min, grid_max );
    clipper::Grid_range unit_cell_box( clipper::Coord_grid(0,0,0), clipper::Coord_grid(out_map->grid_sampling())-clipper::Coord_grid(1,1,1) );
    clipper::Xmap<float>::Map_reference_coord i0, iu, iv, iw;

    clipper::Coord_orth atom_xyz_orth;
    clipper::Coord_frac atom_xyz_frac;
    clipper::Coord_frac trans_xyz_frac;
    clipper::Coord_frac shifted_xyz_frac;
    clipper::Coord_orth shifted_xyz_orth;
    clipper::Coord_orth grid_point_in_orth;
    clipper::Coord_grid grid_min;
    clipper::Coord_grid grid_max;

    float r;
    int count_atoms = 0;
    int count_grid_op = 0;
    for ( unsigned int i=0; i < atoms->size(); i+=3 )
    {
        atom_xyz_orth = clipper::Coord_orth((*atoms)[i], (*atoms)[i+1], (*atoms)[i+2]);
        atom_xyz_frac = atom_xyz_orth.coord_frac(out_map->cell());
        atom_xyz_frac = atom_xyz_frac.lattice_copy_unit();

        for (int k = 0; k < in_map_spacegroup.num_symops(); k++ ) {
            trans_xyz_frac = in_map_spacegroup.symop(k) * atom_xyz_frac;
            trans_xyz_frac = trans_xyz_frac.lattice_copy_unit();
            for (int x_shift=-1; x_shift<2; x_shift++)
            for (int y_shift=-1; y_shift<2; y_shift++)
            for (int z_shift=-1; z_shift<2; z_shift++) {
                count_atoms++;
                shifted_xyz_frac = trans_xyz_frac + clipper::Coord_frac(float(x_shift), float(y_shift), float(z_shift));
                shifted_xyz_orth = shifted_xyz_frac.coord_orth(out_map->cell());

                grid_min = out_map->coord_map(shifted_xyz_orth).coord_grid() + box_min;
                grid_max = out_map->coord_map(shifted_xyz_orth).coord_grid() + box_max;
                i0 = clipper::Xmap<float>::Map_reference_coord((*out_map), grid_min);

                for ( iu = i0; iu.coord().u() <= grid_max.u(); iu.next_u() )
                    for ( iv = iu; iv.coord().v() <= grid_max.v(); iv.next_v() )
                        for ( iw = iv; iw.coord().w() <= grid_max.w(); iw.next_w() )
                            if ( unit_cell_box.in_grid( iw.coord() ) ) {
                                grid_point_in_orth = iw.coord_orth();
                                grid_point_in_orth = grid_point_in_orth - shifted_xyz_orth;
                                r = grid_point_in_orth.lengthsq();
                                (*out_map)[iw] = std::min((*out_map)[iw], r);
                                count_grid_op++;
                            }
                }
            }
    }
    printf("%d %d %d \n", int(atoms->size()/3), count_atoms, count_grid_op);

    set_sqrt(out_map);
}


double Map_point_cluster::volume() const
{
   double cell_vol = this->xmap_ref->cell().volume();
   double n_grid_pts =
      this->xmap_ref->grid_sampling().nu() *
      this->xmap_ref->grid_sampling().nv() *
      this->xmap_ref->grid_sampling().nw();
   double grid_point_vol = cell_vol/n_grid_pts;
   return this->map_grid.size() * grid_point_vol;
}

double Map_point_cluster::density() const
{
   double grid_point_vol = volume();
   return score / grid_point_vol;
}

void Map_point_cluster::add(const clipper::Coord_grid& c_g)
{
  float map_value= this->xmap_ref->get_data(c_g);
  this->map_grid.push_back(c_g);
  this->score += map_value;

  if (map_value > this->max_ponint_score)
  {
    this->max_ponint_score = map_value;
    this->max_point_box_compatible = c_g;
    this->max_point = c_g.unit(this->xmap_ref->grid_sampling());
  }
}

clipper::Coord_orth Map_point_cluster::grid_to_orth(const clipper::Coord_grid& c_g)
{
    clipper::Coord_frac c_f;
    clipper::Coord_orth c_o;
    c_f = c_g.coord_frac(this->xmap_ref->grid_sampling());
    c_o = c_f.coord_orth(this->xmap_ref->cell());
    return c_o;
}

void Map_point_cluster::add_local_max(const clipper::Coord_grid& c_g)
{
    this->map_local_max_orth.push_back(this->grid_to_orth(c_g));
}

void Map_point_cluster::add_skeleton(const clipper::Coord_grid& c_g)
{
    this->skeleton.push_back(this->grid_to_orth(c_g));
}

void Map_point_cluster::add_surface(const clipper::Coord_grid& c_g)
{
    this->surface.push_back(this->grid_to_orth(c_g));
}

clipper::Coord_orth Map_point_cluster::get_min_box_orth()
{
  clipper::Coord_frac f000, f111;
  f000 = get_min_box_frac();
  f111 = get_max_box_frac();

  clipper::Coord_orth p[8];
  p[0] = f000.coord_orth(this->xmap_ref->cell());
  p[1] = f111.coord_orth(this->xmap_ref->cell());
  p[2] = clipper::Coord_frac(f000.u(), f000.v(), f111.w()).coord_orth(this->xmap_ref->cell());
  p[3] = clipper::Coord_frac(f000.u(), f111.v(), f000.w()).coord_orth(this->xmap_ref->cell());
  p[4] = clipper::Coord_frac(f000.u(), f111.v(), f111.w()).coord_orth(this->xmap_ref->cell());
  p[5] = clipper::Coord_frac(f111.u(), f000.v(), f000.w()).coord_orth(this->xmap_ref->cell());
  p[6] = clipper::Coord_frac(f111.u(), f000.v(), f111.w()).coord_orth(this->xmap_ref->cell());
  p[7] = clipper::Coord_frac(f111.u(), f111.v(), f000.w()).coord_orth(this->xmap_ref->cell());

  clipper::Coord_orth min_xyz = p[0];
  for(int i=1; i<8; i++){
      min_xyz[0] = std::min(min_xyz.x(), p[i].x());
      min_xyz[1] = std::min(min_xyz.y(), p[i].y());
      min_xyz[2] = std::min(min_xyz.z(), p[i].z());
  }
  return min_xyz;
}

clipper::Coord_orth Map_point_cluster::get_max_box_orth()
{
  clipper::Coord_frac f000, f111;
  f000 = get_min_box_frac();
  f111 = get_max_box_frac();

  clipper::Coord_orth p[8];
  p[0] = f000.coord_orth(this->xmap_ref->cell());
  p[1] = f111.coord_orth(this->xmap_ref->cell());
  p[2] = clipper::Coord_frac(f000.u(), f000.v(), f111.w()).coord_orth(this->xmap_ref->cell());
  p[3] = clipper::Coord_frac(f000.u(), f111.v(), f000.w()).coord_orth(this->xmap_ref->cell());
  p[4] = clipper::Coord_frac(f000.u(), f111.v(), f111.w()).coord_orth(this->xmap_ref->cell());
  p[5] = clipper::Coord_frac(f111.u(), f000.v(), f000.w()).coord_orth(this->xmap_ref->cell());
  p[6] = clipper::Coord_frac(f111.u(), f000.v(), f111.w()).coord_orth(this->xmap_ref->cell());
  p[7] = clipper::Coord_frac(f111.u(), f111.v(), f000.w()).coord_orth(this->xmap_ref->cell());

  clipper::Coord_orth max_xyz = p[0];
  for(int i=1; i<8; i++){
      max_xyz[0] = std::max(max_xyz.x(), p[i].x());
      max_xyz[1] = std::max(max_xyz.y(), p[i].y());
      max_xyz[2] = std::max(max_xyz.z(), p[i].z());
  }
  return max_xyz;
}

clipper::Coord_orth Map_point_cluster::get_max_point_orth()
{
  return get_max_point_frac().coord_orth(this->xmap_ref->cell());
}

clipper::Coord_orth Map_point_cluster::get_max_point_box_orth()
{
  return get_max_point_box_frac().coord_orth(this->xmap_ref->cell());
}

std::vector<clipper::Coord_orth> Map_point_cluster::get_local_max_orth()
{
  return this->map_local_max_orth;
}

std::vector<clipper::Coord_orth> Map_point_cluster::get_skeleton()
{
  return this->skeleton;
}

std::vector<clipper::Coord_orth> Map_point_cluster::get_surface()
{
  return this->surface;
}

clipper::Coord_frac Map_point_cluster::get_min_box_frac()
{
  clipper::Coord_grid c_min = clipper::Coord_grid(0,0,0);
  if (map_grid.size() <= 0)
    return c_min.coord_frac(this->xmap_ref->grid_sampling());

  std::vector<clipper::Coord_grid>::iterator it = map_grid.begin();
  c_min = (*it);
  for(; it != map_grid.end(); it++)
  {
      clipper::Coord_grid cg = (*it);
      c_min.u() = std::min(c_min.u(), cg.u());
      c_min.v() = std::min(c_min.v(), cg.v());
      c_min.w() = std::min(c_min.w(), cg.w());
  }
  return c_min.coord_frac(this->xmap_ref->grid_sampling());
}

clipper::Coord_frac Map_point_cluster::get_max_box_frac()
{
  clipper::Coord_grid c_min = clipper::Coord_grid(0,0,0);
  if (map_grid.size() <= 0)
    return c_min.coord_frac(this->xmap_ref->grid_sampling());

  std::vector<clipper::Coord_grid>::iterator it = map_grid.begin();
  c_min = (*it);
  for(; it != map_grid.end(); it++)
  {
      clipper::Coord_grid cg = (*it);
      c_min.u() = std::max(c_min.u(), cg.u());
      c_min.v() = std::max(c_min.v(), cg.v());
      c_min.w() = std::max(c_min.w(), cg.w());
  }
  return c_min.coord_frac(this->xmap_ref->grid_sampling());
}

clipper::Coord_frac Map_point_cluster::get_max_point_frac()
{

  return max_point.coord_frac(this->xmap_ref->grid_sampling());
}

clipper::Coord_frac Map_point_cluster::get_max_point_box_frac()
{
  return max_point_box_compatible.coord_frac(this->xmap_ref->grid_sampling());
}

bool Map_point_cluster::is_compact()
{
  if (map_grid.size() <= 1)
  {
    return true;
  }
  std::vector<clipper::Coord_grid> in = std::vector<clipper::Coord_grid>();
  std::vector<clipper::Coord_grid> out = std::vector<clipper::Coord_grid>();

  std::vector<clipper::Coord_grid>::iterator it = map_grid.begin();
  for(; it != map_grid.end(); it++)
  {
      clipper::Coord_grid cg = (*it);
      in.push_back(cg);
  }

  out.push_back(in.back());
  in.pop_back();
  int count_run = 1;
  int count_last = 1;

  std::vector<clipper::Coord_grid>::iterator it_in = in.begin();
  std::vector<clipper::Coord_grid>::iterator it_out = out.begin();
  while (count_run > 0)
  {
    count_run = 0;
    for(it_in=in.begin(); it_in != in.end(); )
    {
      count_last = 0;
      clipper::Coord_grid cg1 = (*it_in);
      for(it_out=out.begin(); it_out != out.end(); it_out++)
      {
          clipper::Coord_grid cg2 = (*it_out);
          int dist = std::max(std::max(std::abs(cg1.u()-cg2.u()), std::abs(cg1.v()-cg2.v())), std::abs(cg1.w()-cg2.w()));
          if (dist == 1)
          {
            out.push_back(cg1);
            count_last += 1;
            break;
          }
      }
      if (count_last==0)
      {
        it_in++;
      }
      else
      {
        count_run += 1;
        it_in = in.erase(it_in);
      }
    }
    //printf("Inte %d %d Size In size %ld Out %ld\n", count_run, count_last, in.size(), out.size());
  }
  return in.size() == 0;
}

bool compare_clusters(const Map_point_cluster &a,
                      const Map_point_cluster &b)
{
   return (a.score > b.score);
};


void find_surface_and_add(
    clipper::Skeleton_basic::Neighbours& surface_neighb,
    clipper::Xmap<float> *in_map,
    Map_point_cluster& mpc,
    clipper::Coord_grid& c_g_start,
    float cut_off)
{
    clipper::Coord_grid c_g;
    for (int i=0; i < surface_neighb.size(); i++)
    {
      c_g = c_g_start + surface_neighb[i];
      if ((*in_map).get_data(c_g) <= cut_off)
      {
        mpc.add_surface(c_g_start);
        break;
      }
    }
}



void find_clusters(
    clipper::Xmap<float> *in_map,
    std::vector<Map_point_cluster> *cluster,
    float cut_off,
    float water_molecule_volume,
    bool verbose)
{
    // if we want to take itself as a neighbour put the second < 0 otherwise 0.05
    // if we want to take all corners put something bigger than 3
    // now we should have all 26 points around
    clipper::Skeleton_basic::Neighbours neighb((*in_map), 0.05, 9.1);
    clipper::Skeleton_basic::Neighbours max_neighb((*in_map), 0.05, 1.7);
    clipper::Skeleton_basic::Neighbours surface_neighb((*in_map), 0.05, 9.1);

    std::cout << "INFO:: Using density cut-off: " << cut_off << std::endl;
    std::cout << "INFO:: Blobs with volume larger than " << water_molecule_volume
              << " A^3 are too big to be considered waters." << std::endl;

    clipper::Xmap_base::Map_reference_index ix;
    std::cout << "INFO:: Finding clusters...";
    std::cout.flush();

    clipper::Xmap<pixel_type> cluster_map = clipper::Xmap<pixel_type>();
    cluster_map.init(in_map->spacegroup(), in_map->cell(),
                     in_map->grid_sampling());

    for (ix = cluster_map.first(); !ix.last(); ix.next())
    {
      cluster_map[ix] = 0;
    }

    std::queue<clipper::Coord_grid> q;
    clipper::Coord_grid c_g;
    clipper::Coord_grid c_g_start;
    bool is_local_max;
    float c_g_start_val;

    for (ix = (*in_map).first(); !ix.last(); ix.next())
    {
      if (! cluster_map[ix])
      {
        if ((*in_map)[ix] > cut_off)
        {
          Map_point_cluster mpc = Map_point_cluster(in_map);
          c_g_start = ix.coord();
          q.push(c_g_start);

          if (!cluster_map.get_data(c_g_start))
          {
            cluster_map.set_data(c_g_start, 1);
            mpc.add(c_g_start);
          }

          while (q.size())
          {
            c_g_start = q.front();
            q.pop();
            for (int i=0; i<neighb.size(); i++)
            {
              c_g = c_g_start + neighb[i];
              if (!cluster_map.get_data(c_g))
              {
                if ((*in_map).get_data(c_g) > cut_off)
                {
                  cluster_map.set_data(c_g, 1);
                  mpc.add(c_g);
                  q.push(c_g);
                }
              }
            }

            // find local maxes
            is_local_max = true;
            c_g_start_val = in_map->get_data(c_g_start);
            for (int i=0; i<max_neighb.size(); i++)
            {
              c_g = c_g_start + max_neighb[i];
              if (in_map->get_data(c_g) > c_g_start_val)
              {
                is_local_max = false;
                break;
              }
            }
            if (is_local_max == true)
            {
              mpc.add_local_max(c_g_start);
            }
            // end is local max


            // uncomment if needed

            // find surface
            //find_surface_and_add(surface_neighb, in_map, mpc, c_g_start, cut_off);
            // end surface
          }
          if (mpc.map_grid.size() > 0 && mpc.volume() > water_molecule_volume)
          {
            cluster->push_back(mpc);
          }
        }
      }
    }
    std::cout << "done" << std::endl;

    std::sort(cluster->begin(), cluster->end(), compare_clusters);

    if (verbose == true)
    {
        unsigned int max_clusters = 6;
        std::cout << "There are " << cluster->size() << " clusers\n";
        std::cout << "Here are the top " << max_clusters << " clusers:\n";
        unsigned int i=0;
        unsigned int cluster_size = std::min((unsigned int)cluster->size(), max_clusters);
        //unsigned int cluster_size = cluster->size();

        while ((i < cluster_size) && ((*cluster)[i].volume() > water_molecule_volume))
        {
          //std::cout << "  Is compact: " << (*cluster)[i].is_compact() << std::endl;
          clipper::Coord_orth bb_min = (*cluster)[i].get_min_box_orth();
          clipper::Coord_orth bb_max = (*cluster)[i].get_max_box_orth();
          clipper::Coord_orth point_max = (*cluster)[i].get_max_point_box_orth();

          double cell_vol = in_map->cell().volume();

          std::cout << "  Cell: a=" << in_map->cell().a()
                           << " b=" << in_map->cell().b()
                           << " c=" << in_map->cell().c()
                           << " alpha="  << in_map->cell().alpha_deg()
                           << " beta="  << in_map->cell().beta_deg()
                           << " gamma="  << in_map->cell().gamma_deg()
                           << " volume=" << cell_vol << std::endl;

          std::cout << "  Grid: u=" << in_map->grid_sampling().nu()
                           << " v=" << in_map->grid_sampling().nv()
                           << " w=" << in_map->grid_sampling().nw() << std::endl;

          std::cout << "  Number: "  << i << " # grid points: "
              << (*cluster)[i].map_grid.size() << " score: "
              << (*cluster)[i].score << " volume: " << (*cluster)[i].volume() << "\n"
              << "  BB Low :     " << bb_min.format() << std::endl
              << "  BB High:     " << bb_max.format() << std::endl
              << "  Max    :     " << point_max.format()
              << std::endl;
          i++;
        }
    }
}


// ====================
// SKELETON
// ====================

int _octree_idx[8][7] = {
    // octant 1
    {0, 1, 3, 4, 9, 10, 12},
    // octant 2
    {1, 4, 10, 2, 5, 11, 13},
    // octant 3
    {3, 4, 12, 6, 7, 14, 15},
    // octant 4
    {4, 5, 13, 7, 15, 8, 16},
    // octant 5
    {9, 10, 12, 17, 18, 20, 21},
    // octant 6
    {10, 11, 13, 18, 21, 19, 22},
    // octant 7
    {12, 14, 15, 20, 21, 23, 24},
    // octant 8
    {13, 15, 16, 21, 22, 24, 25},
};

int _octree_val[8][7][3] = {
    // octant 1
     {{-1, -1, -1}, {2, -1, -1}, {3, -1, -1}, {2, 3, 4}, {5, -1, -1}, {2, 5, 6}, {3, 5, 7}},
    // octant 2
     {{1, -1, -1}, {1, 3, 4}, {1, 5, 6}, {-1, -1, -1}, {4, -1, -1}, {6, -1, -1}, {4, 6, 8}},
    // octant 3
     {{1, -1, -1}, {1, 2, 4}, {1, 5, 7}, {-1, -1, -1}, {4, -1, -1}, {7, -1, -1}, {4, 7, 8}},
    // octant 4
     {{1, 2, 3}, {2, -1, -1}, {2, 6, 8}, {3, -1, -1}, {3, 7, 8}, {-1, -1, -1}, {8, -1, -1}},
    // octant 5
     {{1, -1, -1}, {1, 2, 6}, {1, 3, 7}, {-1, -1, -1}, {6, -1, -1}, {7, -1, -1}, {6, 7, 8}},
    // octant 6
     {{1, 2, 5}, {2, -1, -1}, {2, 4, 8}, {5, -1, -1}, {5, 7, 8}, {-1, -1, -1}, {8, -1, -1}},
    // octant 7
     {{1, 3, 5}, {3, -1, -1}, {3, 4, 8}, {5, -1, -1}, {5, 6, 8}, {-1, -1, -1}, {8, -1, -1}},
    // octant 8
     {{2, 4, 6}, {3, 4, 7}, {4, -1, -1}, {5, 6, 7}, {6, -1, -1}, {7, -1, -1}, {-1, -1, -1}}
};

void octree_labeling(int _oct, int label, pixel_type cube[])
{
    int _idx;
    int _new_octant;

    for (int i = 0; i<7; i++)
    {
        _idx = _octree_idx[_oct-1][i];

        if (cube[_idx] == 1)
        {
            cube[_idx] = label;
            for (int j=0; j<3; j++)
            {
                _new_octant = _octree_val[_oct-1][i][j];
                if (_new_octant >= 0)
                {
                    octree_labeling(_new_octant, label, cube);
                }
            }
        }
    }
}


bool is_simple_point(pixel_type* neighbors)
{
    pixel_type cube[26];
    memcpy(cube, neighbors, 13*sizeof(pixel_type));
    memcpy(cube+13, neighbors+14, 13*sizeof(pixel_type));

    //set initial label
    int label = 2;

    //for all point in the neighborhood
    for (int i=0; i<26; i++){
        if (cube[i] == 1){
            //voxel has not been labeled yet
            //start recursion with any octant that contains the point i
            switch (i){
                case 0:
                case 1:
                case 3:
                case 4:
                case 9:
                case 10:
                case 12:
                    octree_labeling(1, label, cube);
                    break;
                case 2:
                case 5:
                case 11:
                case 13:
                    octree_labeling(2, label, cube);
                    break;
                case 6:
                case 7:
                case 14:
                case 15:
                    octree_labeling(3, label, cube);
                    break;
                case 8:
                case 16:
                    octree_labeling(4, label, cube);
                    break;
                case 17:
                case 18:
                case 20:
                case 21:
                    octree_labeling(5, label, cube);
                    break;
                case 19:
                case 22:
                    octree_labeling(6, label, cube);
                    break;
                case 23:
                case 24:
                    octree_labeling(7, label, cube);
                    break;
                case 25:
                    octree_labeling(8, label, cube);
                    break;
            }

            label += 1;
            if (label - 2 >= 2){
                return false;
            }
        }
    }
    return true;
}


void get_neighborhood(
    clipper::Xmap<pixel_type> &in_xskel, clipper::Coord_grid point, pixel_type* out_neighborhood
)
{
    //Get the neighborhood of a pixel.
    int p, r, c;
    p = point.u();
    r = point.v();
    c = point.w();
    out_neighborhood[0] = in_xskel.get_data(clipper::Coord_grid(p-1, r-1, c-1));
    out_neighborhood[1] = in_xskel.get_data(clipper::Coord_grid(p-1, r,   c-1));
    out_neighborhood[2] = in_xskel.get_data(clipper::Coord_grid(p-1, r+1, c-1));

    out_neighborhood[ 3] = in_xskel.get_data(clipper::Coord_grid(p-1, r-1, c));
    out_neighborhood[ 4] = in_xskel.get_data(clipper::Coord_grid(p-1, r,   c));
    out_neighborhood[ 5] = in_xskel.get_data(clipper::Coord_grid(p-1, r+1, c));

    out_neighborhood[ 6] = in_xskel.get_data(clipper::Coord_grid(p-1, r-1, c+1));
    out_neighborhood[ 7] = in_xskel.get_data(clipper::Coord_grid(p-1, r,   c+1));
    out_neighborhood[ 8] = in_xskel.get_data(clipper::Coord_grid(p-1, r+1, c+1));

    out_neighborhood[ 9] = in_xskel.get_data(clipper::Coord_grid(p, r-1, c-1));
    out_neighborhood[10] = in_xskel.get_data(clipper::Coord_grid(p, r,   c-1));
    out_neighborhood[11] = in_xskel.get_data(clipper::Coord_grid(p, r+1, c-1));

    out_neighborhood[12] = in_xskel.get_data(clipper::Coord_grid(p, r-1, c));
    out_neighborhood[13] = in_xskel.get_data(clipper::Coord_grid(p, r,   c));
    out_neighborhood[14] = in_xskel.get_data(clipper::Coord_grid(p, r+1, c));

    out_neighborhood[15] = in_xskel.get_data(clipper::Coord_grid(p, r-1, c+1));
    out_neighborhood[16] = in_xskel.get_data(clipper::Coord_grid(p, r,   c+1));
    out_neighborhood[17] = in_xskel.get_data(clipper::Coord_grid(p, r+1, c+1));

    out_neighborhood[18] = in_xskel.get_data(clipper::Coord_grid(p+1, r-1, c-1));
    out_neighborhood[19] = in_xskel.get_data(clipper::Coord_grid(p+1, r,   c-1));
    out_neighborhood[20] = in_xskel.get_data(clipper::Coord_grid(p+1, r+1, c-1));

    out_neighborhood[21] = in_xskel.get_data(clipper::Coord_grid(p+1, r-1, c));
    out_neighborhood[22] = in_xskel.get_data(clipper::Coord_grid(p+1, r,   c));
    out_neighborhood[23] = in_xskel.get_data(clipper::Coord_grid(p+1, r+1, c));

    out_neighborhood[24] = in_xskel.get_data(clipper::Coord_grid(p+1, r-1, c+1));
    out_neighborhood[25] = in_xskel.get_data(clipper::Coord_grid(p+1, r,   c+1));
    out_neighborhood[26] = in_xskel.get_data(clipper::Coord_grid(p+1, r+1, c+1));
}


bool is_endpoint(pixel_type* neighbors){
    //An endpoint has exactly one neighbor in the 26-neighborhood.
    unsigned int s = 0;
    for (int j=0; j<27; j++)
    {
        s += neighbors[j];
    }
    return s == 2;
}


int Euler_LUT[256] = { 0,  1,  0, -1,  0, -1,  0,  1,  0, -3,  0, -1,  0, -1,  0,  1,  0,
                      -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0, -3,
                       0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0, -1,  0,
                       1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0, -3,  0,  3,
                       0, -1,  0,  1,  0,  1,  0,  3,  0, -1,  0,  1,  0, -1,  0,  1,  0,
                       1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0,  1,  0,  3,  0,  3,
                       0,  1,  0,  5,  0,  3,  0,  3,  0,  1,  0, -1,  0,  1,  0,  1,  0,
                      -1,  0,  3,  0,  1,  0,  1,  0, -1,  0, -7,  0, -1,  0, -1,  0,  1,
                       0, -3,  0, -1,  0, -1,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,
                       3,  0,  1,  0,  1,  0, -1,  0, -3,  0, -1,  0,  3,  0,  1,  0,  1,
                       0, -1,  0,  3,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,
                       1,  0,  1,  0, -1,  0, -3,  0,  3,  0, -1,  0,  1,  0,  1,  0,  3,
                       0, -1,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,
                       1,  0, -1,  0,  1,  0,  3,  0,  3,  0,  1,  0,  5,  0,  3,  0,  3,
                       0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1};


// Fill the look-up table for indexing octants for computing the Euler
// characteristic. See is_Euler_invariant routine below.
int _neighb_idx[8][7] = {
    {2, 1, 11, 10, 5, 4, 14},      // NEB
    {0, 9, 3, 12, 1, 10, 4},       // NWB
    {8, 7, 17, 16, 5, 4, 14},      // SEB
    {6, 15, 7, 16, 3, 12, 4},      // SWB
    {20, 23, 19, 22, 11, 14, 10},  // NEU
    {18, 21, 9, 12, 19, 22, 10},   // NWU
    {26, 23, 17, 14, 25, 22, 16},  // SEU
    {24, 25, 15, 16, 21, 22, 12},  // SWU
};


bool is_Euler_invariant(pixel_type* neighbors)
{
    //Check if a point is Euler invariant.
    //Calculate Euler characteristic for each octant and sum up.

    int n, _idx, euler_char = 0;
    for (int i=0; i<8; i++)
    {
        n = 1;
        for (int j=0; j<7; j++)
        {
            _idx = _neighb_idx[i][j];
            if (neighbors[_idx] == 1)
            {
                n |= 1 << (7 - j);
            }
        }
        euler_char += Euler_LUT[n];
    }
    return euler_char == 0;
}


std::vector<clipper::Coord_grid> find_simple_point_candidates(
    clipper::Xmap<pixel_type> &in_out_xskel, Map_point_cluster &cluster, int curr_border)
{
    std::vector<clipper::Coord_grid> simple_border_points = std::vector<clipper::Coord_grid>();

    pixel_type neighborhood[27];
    int p, r, c;
    bool is_border_pt;
    clipper::Coord_grid point;
    int xskel_val;

    // loop through the image
    // NB: each loop is from 1 to size-1: img is padded from all sides
    //-clipper::Xmap_base::Map_reference_index ix;
    //-for (ix = in_out_xskel.first(); !ix.last(); ix.next())
    std::vector<clipper::Coord_grid>::iterator point_it;
    for (point_it = cluster.map_grid.begin(); point_it != cluster.map_grid.end(); point_it++)
    {
        point = (*point_it);
        p = point.u();
        r = point.v();
        c = point.w();

        xskel_val = in_out_xskel.get_data(point);

        if (xskel_val != 1){
            continue;
        }

        is_border_pt = ((curr_border == 1 && in_out_xskel.get_data(clipper::Coord_grid(p, r, c-1)) == 0) ||  //N
                        (curr_border == 2 && in_out_xskel.get_data(clipper::Coord_grid(p, r, c+1)) == 0) ||  //S
                        (curr_border == 3 && in_out_xskel.get_data(clipper::Coord_grid(p, r+1, c)) == 0) ||  //E
                        (curr_border == 4 && in_out_xskel.get_data(clipper::Coord_grid(p, r-1, c)) == 0) ||  //W
                        (curr_border == 5 && in_out_xskel.get_data(clipper::Coord_grid(p+1, r, c)) == 0) ||  //U
                        (curr_border == 6 && in_out_xskel.get_data(clipper::Coord_grid(p-1, r, c)) == 0));   //B

        // current point is not deletable
        if (!is_border_pt){
            continue;
        }

        get_neighborhood(in_out_xskel, point, neighborhood);

        // check if (p, r, c) can be deleted:
        // * it must not be an endpoint;
        // * it must be Euler invariant (condition 1 in [Lee94]_); and
        // * it must be simple (i.e., its deletion does not change
        //   connectivity in the 3x3x3 neighborhood)
        //   this is conditions 2 and 3 in [Lee94]_
        if (is_endpoint(neighborhood) ||
            !is_Euler_invariant(neighborhood) ||
            !is_simple_point(neighborhood))
        {
            continue;
        }

        simple_border_points.push_back(point);
    }

    return simple_border_points;
}


void find_skeleton_custom(
    clipper::Xmap<pixel_type> &in_out_xskel,
    Map_point_cluster &cluster,
    bool verbose)
{
    unsigned int unchanged_borders = 0;
    int curr_border;
    unsigned int num_borders = 6;
    int borders[6] = {4, 3, 2, 1, 5, 6};
    bool no_change;
    pixel_type neighb[27];

    clipper::Coord_grid point;
    std::vector<clipper::Coord_grid>::iterator point_it;
    std::vector<clipper::Coord_grid> simple_border_points;

    while (unchanged_borders < num_borders)
    {
        unchanged_borders = 0;
        for (unsigned int j=0; j<num_borders; j++){
            curr_border = borders[j];
            simple_border_points = find_simple_point_candidates(in_out_xskel, cluster, curr_border);

            // sequential re-checking to preserve connectivity when deleting
            // in a parallel way
            no_change = true;
            for (point_it = simple_border_points.begin(); point_it != simple_border_points.end(); point_it++)
            {
               point = (*point_it);
               get_neighborhood(in_out_xskel, point, neighb);
               if (is_simple_point(neighb) == true)
               {
                    in_out_xskel.set_data(point, 0);
                    no_change = false;
               }
            }
            if (no_change == true){
                unchanged_borders += 1;
            }
        }
    }
}



void find_skeleton(
    clipper::Xmap<float> *in_map,
    std::vector<Map_point_cluster> *in_out_cluster,
    float cut_off,
    bool verbose
)
{
    // init temp map
    clipper::Xmap<pixel_type> xskel = clipper::Xmap<pixel_type>();
    xskel.init(in_map->spacegroup(), in_map->cell(), in_map->grid_sampling());

    // clean xskel
    clipper::Xmap_base::Map_reference_index ix;
    for (ix = xskel.first(); !ix.last(); ix.next())
    {
        xskel[ix] = 0;
    }

    int count = 0;
    std::vector<Map_point_cluster>::iterator cluster_it;
    std::vector<clipper::Coord_grid>::iterator grid_point_it;

    // for each cluster
    for (cluster_it = in_out_cluster->begin(); cluster_it != in_out_cluster->end(); cluster_it++)
    {
        if (verbose)
        {
            std::cout << "SKEL:: Blob " << count << " " << (*cluster_it).volume() << " " << (*cluster_it).score << std::endl;
            count++;
            std::cout.flush();
        }

        // assign 1 to grid point in xskel found in cluster
        for (grid_point_it = cluster_it->map_grid.begin(); grid_point_it != cluster_it->map_grid.end(); grid_point_it++)
        {
            xskel.set_data((*grid_point_it), 1);
        }

        if (verbose)
        {
            std::cout << "SKEL:: Ready to calculate " << std::endl;
        }

        // here calculate
        find_skeleton_custom(xskel, (*cluster_it), verbose);

        // pixel type should be int to run that
        //clipper::Skeleton_basic skeleton = clipper::Skeleton_basic(xskel, (*in_map), 2);
        // or that
        //clipper::Skeleton_fast<int, float> skeleton = clipper::Skeleton_fast<int, float>(xskel, (*in_map));

        // end calculate

        // add and clean
        // assign 0 to grid point in xskel found in cluster
        for (grid_point_it = cluster_it->map_grid.begin(); grid_point_it != cluster_it->map_grid.end(); grid_point_it++)
        {
            if (xskel.get_data((*grid_point_it)) > 0)
            {
                cluster_it->add_skeleton((*grid_point_it));
            }

            xskel.set_data((*grid_point_it), 0);
        }

        if (verbose)
        {
            std::cout.flush();
        }
    }
}

