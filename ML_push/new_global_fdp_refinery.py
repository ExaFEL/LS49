from __future__ import print_function, division
from six.moves import range
from six.moves import cPickle as pickle
from scitbx.array_family import flex
import scitbx
import math

# %%% boilerplate specialize to packaged big data %%%
import os
from LS49.sim import step5_pad
from LS49.sim import step4_pad
from LS49.spectra import generate_spectra
ls49_big_data = os.environ["LS49_BIG_DATA"] # get absolute path from environment
step5_pad.big_data = ls49_big_data
step4_pad.big_data = ls49_big_data
generate_spectra.big_data = ls49_big_data
# %%%%%%
from LS49.sim.step5_pad import data
local_data = data()
Fe_oxidized_model = local_data.get("Fe_oxidized_model")
Fe_reduced_model = local_data.get("Fe_reduced_model")
Fe_metallic_model = local_data.get("Fe_metallic_model")

def get_static_fcalcs_with_HKL_lookup():
    from LS49.work2_for_aca_lsq.remake_range_intensities_with_complex import get_C2_structures, gen_fmodel_with_complex
    C2_structures = get_C2_structures()

    energy = 7070.0
    GF_whole7070 = gen_fmodel_with_complex.from_structure(C2_structures[0],energy
                   ).from_parameters(algorithm="fft")
    GF_whole7070 = GF_whole7070.as_P1_primitive()
    f_container7070 = GF_whole7070.get_fmodel()
    Fmodel_whole7070 = f_container7070.f_model
    Fmodel_indices = Fmodel_whole7070.indices() # common structure defines the indices
    F_bulk = f_container7070.fmodel.arrays.core.data.f_bulk
    F_bulk.reshape(flex.grid((Fmodel_indices.size(),1))) # in-place reshape, non-standard

    result = flex.complex_double(flex.grid((Fmodel_indices.size(),100)))

    # common structure to represent the wavelength-dependent non-Fe diffraction (bulk+atoms)
    for incr in range(100):
      energy = 7070.5 + incr

      GF_non_Fe = gen_fmodel_with_complex.from_structure(C2_structures[1],energy
                  ).from_parameters(algorithm="fft")
      GF_non_Fe = GF_non_Fe.as_P1_primitive()
      f_container = GF_non_Fe.get_fmodel()
      Fcalc_non_Fe = f_container.fmodel.f_calc().data()
      Fcalc_non_Fe.reshape(flex.grid((Fmodel_indices.size(),1))) # in-place reshape, non-standard

      result.matrix_paste_block_in_place((F_bulk + Fcalc_non_Fe),0,incr)

    HKL_lookup = {}
    for iw in range(len(Fmodel_indices)):
      HKL_lookup[Fmodel_indices[iw]] = iw

    # result holds a table of complex double structure factors.  Rows are Miller indices H.
    # columns are F_H(energy, 100 channels) for F(bulk) + F(non-Fe atoms). Thus this is
    # the energy-dependent portion of the calculation that is not dependent on the iron model.
    return HKL_lookup, result

#specialize this file to look at one particular index
distance_mm = 141.7
pixel_sz_mm = 0.11
mos_rotation_deg = 0.05

def get_items(myrank,N_total,N_stride):
  for key in range(N_total):
    #each rank should only allow keys in the range from myrank*N_stride to (myrank+1)*N_stride
    if key<myrank*N_stride: continue
    if key >= (myrank+1) * N_stride: continue
    if key >= N_total : continue
    try:
      with open("abc_coverage/abcX%06d.pickle"%key,"rb") as F:
        T = pickle.load(F)
    except IOError:
      #print("No file abc_coverage/abcX%06d.pickle"%key)
      continue
    yield T,key
def pprint(M):
  islow,ifast=M.focus()
  for x in range(islow):
    print (" ".join([("%4.2f"%(10*M[(x,y)])) for y in range(ifast)]))

"""
Road map from here.
1) Get this checked in to github
2) Make new class to do more things with Fmodel.
OLD = confirm_P1_range_reduced_intensities_dict.pickle
NEW = same calculation, (P1 structure factor intensities for every energy)
      but with terms separated out for bulk (Fmodel-Fcalc), atoms except Fe (Fcalc), Fe1(Fcalc), Fe2(Fcalc)
Express model(OLD) as a vector sum of NEW
Verify that (Fsq)OLD==(Fsq)NEW to close approx.
3) Verify that Fbulk doesn't change (much) as a function of energy
3) Verify that F(non-Fe) does change.  It should:  over 100 eV.
3) Express Fm in terms of F0+fp+fdp.  Get derivatives of F with respect to all parameters.
4) Change the interface so I can evaluate the structure factors with current self.x parameter set.
"""

class fit_one_image_multispot:
  def __init__(self,list_of_images,HKL_lookup,model_intensities):
    import scitbx
    #lay out the parameters.
    self.n_spots = len(list_of_images)
    self.n = 3*self.n_spots + 1
    self.x = flex.double()
    for ispot in range(self.n_spots):
      self.x.append(list_of_images[ispot].bkgrd_a[0])
      self.x.append(list_of_images[ispot].bkgrd_a[1])
      self.x.append(list_of_images[ispot].bkgrd_a[2])
    self.x.append(1.)
    self.roi_model_pixels = []
    for ispot in range(self.n_spots):
      intensity = list_of_images[ispot].simtbx_intensity_7122
      this_P1_Miller_index = list_of_images[ispot].simtbx_P1_miller
      lookup_idx = HKL_lookup[this_P1_Miller_index]
      energy_dependent_intensity = model_intensities.matrix_copy_block(
        i_row=lookup_idx,i_column=0,n_rows=1,n_columns=100)
      rescale_factor = energy_dependent_intensity.as_1d() / intensity
      channels = list_of_images[ispot].channels
      self.roi_model_pixels.append(rescale_factor[0] * channels[0])

      for ichannel in range(1,len(channels)):
        # rescale_factor[51] always == 1, equivalent to simtbx_intensity_7122
        self.roi_model_pixels[ispot] += rescale_factor[2*ichannel] * channels[2*ichannel]
    self.sb_data = []
    for ispot in range(self.n_spots):
      self.sb_data.append(list_of_images[ispot].sb_data)
    self.minimizer = scitbx.lbfgs.run(target_evaluator=self,
        termination_params=scitbx.lbfgs.termination_parameters(
        traditional_convergence_test=True,
        traditional_convergence_test_eps=1.e-3, #significantly (4x) quicker than 1.e-4
        #drop_convergence_test_max_drop_eps=max_drop_eps,
        #min_iterations=min_iterations,
        #max_iterations = None,
        max_calls=1000)
    )
    self.a = self.x

  def print_step(self,message,target):
    print ("%s %10.4f"%(message,target),
           "["," ".join(["%10.4f"%a for a in self.x]),"]")

  def compute_functional_and_gradients(self):
    self.a = self.x
    f = 0.;
    g = flex.double(self.n)
    for ispot in range(self.n_spots):
      F = self.sb_data[ispot].focus()
      for x in range(F[1]):
        for y in range(F[2]):
          model_lambda = self.a[3*ispot+0]*x+self.a[3*ispot+1]*y+self.a[3*ispot+2]+ \
                         self.a[-1]*self.roi_model_pixels[ispot][x,y]
          if model_lambda<=0:
            f+= model_lambda # complete kludge, guard against math domain error
          else:
            datapt = self.sb_data[ispot][0,x,y]
            f += model_lambda - datapt * math.log(model_lambda)
          g[3*ispot+0] += x * (1. - datapt/model_lambda) # from handwritten notes
          g[3*ispot+1] += y * (1. - datapt/model_lambda)
          g[3*ispot+2] += (1. - datapt/model_lambda)
          g[-1] += self.roi_model_pixels[ispot][x,y] * (1. - datapt/model_lambda)
    #self.print_step("LBFGS stp",f)
    return f, g

from LS49.sim.fdp_plot import george_sherrell
class george_sherrell_star(george_sherrell):
  def __init__(self,fp,fdp):
    self.energy = flex.double([7071. + incr for incr in range(100)])
    self.fp = fp
    self.fdp = fdp

class rank_0_fit_all_f:
  def __init__(self,FE1_model=Fe_oxidized_model,FE2_model=Fe_reduced_model):
    self.n = 400
    self.x = flex.double(self.n)    #lay out the parameters.
    for incr in range(100):
      energy = 7070.5 + incr
      wavelength = 12398.425/energy
      newfp,newfdp = FE1_model.fp_fdp_at_wavelength(angstroms=wavelength)
      self.x[incr]=newfp; self.x[incr+100]=newfdp
      newfp,newfdp = FE2_model.fp_fdp_at_wavelength(angstroms=wavelength)
      self.x[incr+200]=newfp; self.x[incr+300]=newfdp

  def reinitialize(self, rank, per_rank_items, per_rank_keys, per_rank_G,
                   HKL_lookup, static_fcalcs, model_intensities):
    from libtbx import adopt_init_args
    adopt_init_args(self, locals())
    self.model_intensities_reinitialized_for_these_parameters = True
    self.starting_params_cached = False
    self.iteration = 0

  def plot_em(self):
    if not self.starting_params_cached:
      from matplotlib import pyplot as plt
      self.plt = plt
      self.plt.ion() # interactive - on
    self.plt.cla() #clear last access

    # ground truth
    from LS49.sim.step5_pad import full_path
    GS = george_sherrell(full_path("data_sherrell/pf-rd-ox_fftkk.out"))
    GS.plot_them(self.plt,f1="b-",f2="b-")
    GS = george_sherrell(full_path("data_sherrell/pf-rd-red_fftkk.out"))
    GS.plot_them(self.plt,f1="r-",f2="r-")
    GS = george_sherrell(full_path("data_sherrell/Fe_fake.dat")) # with interpolated points
    GS.plot_them(self.plt,f1="m-",f2="m-")

    # starting values
    if not self.starting_params_cached:
      self.starting_params_FE1 = self.x[0:200]
      self.starting_params_FE2 = self.x[200:400]
      self.starting_params_cached = True
    GS = george_sherrell_star(fp = self.starting_params_FE1[0:100],fdp = self.starting_params_FE1[100:200])
    GS.plot_them(self.plt,f1="bx",f2="bx")
    GS = george_sherrell_star(fp = self.starting_params_FE2[0:100],fdp = self.starting_params_FE2[100:200])
    GS.plot_them(self.plt,f1="rx",f2="rx")

    # current values
    GS = george_sherrell_star(fp = self.x[0:100],fdp = self.x[100:200])
    GS.plot_them(self.plt,f1="b.",f2="b.")
    GS = george_sherrell_star(fp = self.x[200:300],fdp = self.x[300:400])
    GS.plot_them(self.plt,f1="r.",f2="r.")

    self.plt.axes().set_xlim((7088,7152))
    self.plt.axes().set_ylim((-8.3,4.2))
    self.plt.title("Iteration %d"%self.iteration)
    self.plt.draw()
    self.plt.pause(0.2)
    #self.plt.show()

  def compute_functional_and_gradients(self):
    a = self.x
    f = 0.;
    g = flex.double(self.n)
    self.iteration += 1
    print("inside compute rank",self.rank,"for iteration",self.iteration)
    if self.rank==0:  self.plot_em()

    #Should plot the first two function calls (rank0) before failing on assert
    #assert self.model_intensities_reinitialized_for_these_parameters
    if not self.model_intensities_reinitialized_for_these_parameters:
      # recalculate model intensities for each target evaluation
      if True: #self.rank == 0:

        FE1 = george_sherrell_star(fp = self.x[0:100],fdp = self.x[100:200])
        FE2 = george_sherrell_star(fp = self.x[200:300],fdp = self.x[300:400])
        from LS49.work2_for_aca_lsq.remake_range_intensities_with_complex \
           import get_intensity_structure
        self.model_intensities = get_intensity_structure(
           self.static_fcalcs,FE1_model=FE1,FE2_model=FE2)

        #transmitted_info = model_intensities
      else:
        transmitted_info = None
      #from libtbx.mpi4py import MPI
      #comm = MPI.COMM_WORLD
      #self.model_intensities = comm.bcast(transmitted_info, root = 0)
      #comm.barrier()

    this_rank_N_images = len(self.per_rank_keys)
    for i_image in range(this_rank_N_images):
      this_image = self.per_rank_items[i_image]
      this_image_N_spots = len(this_image)
      this_G = self.per_rank_G[i_image]
      for i_spot in range(this_image_N_spots):
        this_spot = this_image[i_spot]
        miller_index = this_spot.simtbx_P1_miller
        this_ref_intensity = this_spot.simtbx_intensity_7122

        lookup_idx = self.HKL_lookup[miller_index]
        energy_dependent_intensity = self.model_intensities.matrix_copy_block(
                      i_row=lookup_idx,i_column=0,n_rows=1,n_columns=100)
        energy_dependent_derivatives = self.model_intensities.matrix_copy_block(
                      i_row=lookup_idx,i_column=100,n_rows=1,n_columns=400)
        rescale_factor = energy_dependent_intensity.as_1d() / this_ref_intensity
        channels = this_spot.channels
        roi_model_pixels = rescale_factor[0] * channels[0]
        for ichannel in range(1,len(channels)):
          roi_model_pixels += rescale_factor[2*ichannel] * channels[2*ichannel]

        abc = this_spot.bkgrd_a
        F = this_spot.roi.focus()
        for x in range(F[0]):
          for y in range(F[1]):
            bkgrd = abc[0]*x+abc[1]*y+abc[2]
            model_lambda = bkgrd + this_G * roi_model_pixels[x,y]
            if model_lambda<=0:
              f+= model_lambda # complete kludge, guard against math domain error
              datapt = 0.
            else:
              datapt = this_spot.sb_data[0,x,y]
              f += model_lambda - datapt * math.log(model_lambda)
              g_independent_prefactor = (this_G/this_ref_intensity)*(1. - datapt/model_lambda)
              for ichannel in range(len(channels)):
                g_channel_prefactor = g_independent_prefactor * channels[2*ichannel][x,y]
                for param_type in range(4):
                  derivative_address = 2*ichannel + 100*param_type
                  g[derivative_address] += g_channel_prefactor * \
                                           energy_dependent_derivatives[derivative_address]
            #print (x,y,bkgrd,model_lambda,datapt,int(model_lambda - datapt))

    self.model_intensities_reinitialized_for_these_parameters = False
    if self.rank == 0: self.print_step("LBFGS Iteration %d"%self.iteration,f,g)
    return f, g
  def print_step(self,message,target,g):
    print ("%s %10.4f"%(message,target),
           "["," ".join(["%10.4f"%a for a in g]),"]")

class MPI_Run(object):
  def __init__(self):
    from xfel.merging.application.mpi_helper import mpi_helper
    self.mpi_helper = mpi_helper()

  def __del__(self):
    self.mpi_helper.finalize()

  def parse_input(self):
    '''Parse input at rank 0 and broadcast the input parameters and options to all ranks'''

    if self.mpi_helper.rank == 0:
      from LS49.ML_push.phil import phil_scope
      help_message = '''Refine fp fdp parameters.'''

      # The script usage
      import libtbx.load_env
      self.usage = "usage: %s [options] [param.phil] " % libtbx.env.dispatcher_name
      self.parser = None

      '''Initialize the script.'''
      from dials.util.options import OptionParser
      # Create the parser
      self.parser = OptionParser(
        usage=self.usage,
        phil=phil_scope,
        epilog=help_message)

      # Parse the command line. quick_parse is required for MPI compatibility
      params, options = self.parser.parse_args(show_diff_phil=True,quick_parse=True)

      # prepare for transmitting input parameters to all ranks
      transmitted = dict(params = params, options = options)
    else:
      transmitted = None

    # broadcast parameters and options to all ranks

    transmitted = self.mpi_helper.comm.bcast(transmitted, root = 0)

    self.params = transmitted['params']
    self.options = transmitted['options']

  def run(self):

    self.parse_input()

    N_total = self.params.N_total # number of items to simulate, nominally 100000
    logical_rank = self.mpi_helper.rank
    logical_size = self.mpi_helper.size
    if self.mpi_helper.rank==0 and self.mpi_helper.size==1: # special case of testing it
      try:
        logical_rank = self.params.tester.rank
        logical_size = self.params.tester.size
      except Exception: pass
    N_stride = int(math.ceil(N_total/logical_size)) # total number of tasks per rank
    print ("hello from rank %d of %d with stride %d"%(logical_rank,logical_size,N_stride))

    from scitbx.lbfgs.tst_mpi_split_evaluator import mpi_split_evaluator_run
    from scitbx.lbfgs.tst_mpi_split_evaluator import run_mpi as simple_tester
    simple_tester()

    if self.params.starting_model.algorithm=="to_file":
      if self.mpi_helper.rank == 0:
        HKL_lookup,static_fcalcs = get_static_fcalcs_with_HKL_lookup()
        from LS49.work2_for_aca_lsq.remake_range_intensities_with_complex \
           import get_intensity_structure
        model_intensities = get_intensity_structure(
           static_fcalcs,FE1_model=Fe_oxidized_model,FE2_model=Fe_reduced_model)
        with (open(self.params.starting_model.filename,"wb")) as out:
          pickle.dump(HKL_lookup,out,pickle.HIGHEST_PROTOCOL)
          pickle.dump(static_fcalcs,out,pickle.HIGHEST_PROTOCOL)
          pickle.dump(model_intensities,out,pickle.HIGHEST_PROTOCOL)
      return

def run_mpi():

  if rank == 0:
    print ("set up in rank 0")
    #get initial model
    if False: # generate initial conditions for the first time
      HKL_lookup,static_fcalcs = get_static_fcalcs_with_HKL_lookup()
      from LS49.work2_for_aca_lsq.remake_range_intensities_with_complex \
           import get_intensity_structure
      model_intensities = get_intensity_structure(
           static_fcalcs,FE1_model=Fe_oxidized_model,FE2_model=Fe_reduced_model)
      with (open("new_global_fdp_big_data.pickle","wb")) as out:
        pickle.dump(HKL_lookup,out,pickle.HIGHEST_PROTOCOL)
        pickle.dump(static_fcalcs,out,pickle.HIGHEST_PROTOCOL)
        pickle.dump(model_intensities,out,pickle.HIGHEST_PROTOCOL)
    else:
      with (open("new_global_fdp_big_data.pickle","rb")) as inp:
        HKL_lookup = pickle.load(inp)
        static_fcalcs = pickle.load(inp)
        model_intensities = pickle.load(inp)

    transmitted_info = dict(HKL_lookup = HKL_lookup,
        static_fcalcs = static_fcalcs, model_intensities = model_intensities
    )
    print ("finished setup in rank 0")
  else:
    transmitted_info = None
  transmitted_info = comm.bcast(transmitted_info, root = 0)
  comm.barrier()
  per_rank_items = []
  per_rank_keys = []
  per_rank_G = []
  min_spots = 3
  N_input=0
  for item,key in get_items(logical_rank,N_total,N_stride):
    N_input+=1
    if len(item) >= min_spots:
      per_rank_items.append(item)
      per_rank_keys.append(key)
      FOI = fit_one_image_multispot(list_of_images=item,
            HKL_lookup = transmitted_info["HKL_lookup"],
            model_intensities = transmitted_info["model_intensities"])

      print ("""LLG Image %06d on %d Bragg spots NLL    channels F = %9.1f"""%(
        key, len(item), FOI.compute_functional_and_gradients()[0]))
      # put the newly refined background model back into the item
      for ihkl in range(FOI.n_spots):
        per_rank_items[-1][ihkl].bkgrd_a = flex.double(
                    [FOI.a[3*ihkl+0],FOI.a[3*ihkl+1],FOI.a[3*ihkl+2]])
      per_rank_G.append( FOI.a[-1] )

  print ("rank %d has %d refined images"%(logical_rank,len(per_rank_items)))
  N_ranks = comm.reduce(1, MPI.SUM, 0)
  N_refined_images = comm.reduce(len(per_rank_items), MPI.SUM, 0)
  N_input_images = comm.reduce(N_input, MPI.SUM, 0)
  comm.barrier()
  if rank==0:
    print ("final report %d ranks, %d input images, %d refined models"%(
    N_ranks, N_input_images, N_refined_images))
    print ("Finished finding initial G and abc factors")
    print ("Initiating the full minimization")

  W = rank_0_fit_all_f()
  W.reinitialize(rank, per_rank_items, per_rank_keys, per_rank_G, transmitted_info["HKL_lookup"],
                 transmitted_info["static_fcalcs"],transmitted_info["model_intensities"])

  minimizer = mpi_split_evaluator_run(target_evaluator=W,
        termination_params=scitbx.lbfgs.termination_parameters(
        traditional_convergence_test=True,
        traditional_convergence_test_eps=1.e-2,
        max_calls=10)
      )
  if rank==0:
    print("Minimizer ended at iteration",W.iteration)
    for ir in range(0):
      raw_input("press return %d of 10..."%ir)
    #W.plt.ioff() # interactive off, preserve screen display
  comm.barrier()
  """
Actual work on Dec. 21
1) Try to instrument get_intensities to develop plan for making it 10x more efficient.
---
multi-rank MPI doesn't seem to work.
Try disabling maptplotib (use a flag)
Try doing work on dials; avoid necessity for shifter; avoid lustre file system
Try refactoring get_intensities in case there is a problem with workers timing out
Try using only two ranks
Try turning off any try....except loops in lbfgs
Try removing the zero-gradient parameters
Is there a way to spoof get_intensities so it skips the calculation?
Or a way to put it into C++ so it goes much faster?
Is it using up too much memory?
0) possibly crucial is the p(model) smoothing restraint
1) confirm that we can get further than iteration 2 with srun 32
2) scale up to 100000?
3) scale up to 1024 cores?
4) refactor get_intensities to use MPI?
"""
  #from IPython import embed; embed()
  """once done with all this, try
  1) starting points with all four pre-set ox/red states
  2) use 100 channels instead of 50
  3) use macrocycle over abcG / fdp refinement
  """
if __name__=="__main__":
  # 242 sec.
  Usage = """srun -n 32 -c 2 libtbx.python new_global_fdp_refinery.py #smallish test case, 1 node
             libtbx.python new_global_fdp_refinery.py 45
             ...either works only under: salloc -C haswell -N1 -q interactive -t 04:00:00
  """

  script = MPI_Run()
  result = script.run()
