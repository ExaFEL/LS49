from __future__ import division, print_function
from six.moves import range
from scitbx.array_family import flex
from scitbx.matrix import sqr
import libtbx.load_env # possibly implicit
from cctbx import crystal
from time import time
from omptbx import omp_get_num_procs

# %%% boilerplate specialize to packaged big data %%%
import os
from LS49.adse13_161 import step5_pad
from LS49.sim import step4_pad
from LS49.spectra import generate_spectra
ls49_big_data = os.environ["LS49_BIG_DATA"] # get absolute path from environment
step5_pad.big_data = ls49_big_data
step4_pad.big_data = ls49_big_data
generate_spectra.big_data = ls49_big_data

from LS49.sim.util_fmodel import gen_fmodel
from LS49.adse13_161.step5_pad import data
# %%%%%%

# Develop procedure for MPI control

def tst_one(image,spectra_iter,crystal,random_orientation,sfall_channels):

  quick = False
  if quick: prefix_root="step5_batch_%06d"
  else: prefix_root="step5_MPIbatch_%06d"

  file_prefix = prefix_root%image
  rand_ori = sqr(random_orientation)
  
  from LS49.adse13_161.step5_pad import run_sim2smv
  run_sim2smv(prefix = file_prefix,crystal = crystal,spectra_iter=spectra_iter,rotation=rand_ori,quick=quick,rank=rank, sfall_main=sfall_main, local_data=local_data, sfall_channels=sfall_channels)

if __name__=="__main__":
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  import omptbx
  workaround_nt = int(os.environ.get("OMP_NUM_THREADS",1))
  omptbx.omp_set_num_threads(workaround_nt)
  N_total = int(os.environ["N_SIM"]) # number of items to simulate
  N_stride = size # total number of worker tasks
  #print("hello from rank %d of %d"%(rank,size),"with omp_threads=",omp_get_num_procs())
  import datetime
  start_elapse = time()
  if rank == 0:
    print("Rank 0 time", datetime.datetime.now())
    from LS49.spectra.generate_spectra import spectra_simulation
    from LS49.adse13_161.step5_pad import microcrystal
    #print("hello2 from rank %d of %d"%(rank,size))
    SS = spectra_simulation()
    C = microcrystal(Deff_A = 4000, length_um = 4., beam_diameter_um = 1.0) # assume smaller than 10 um crystals
    mt = flex.mersenne_twister(seed=0)
    random_orientations = []
    for iteration in range(N_total):
      random_orientations.append( mt.random_double_r3_rotation_matrix() )
    transmitted_info = dict(spectra = SS,
                            crystal = C,
                            random_orientations = random_orientations)
  else:
    transmitted_info = None
  transmitted_info = comm.bcast(transmitted_info, root = 0)
  comm.barrier()

  # MU *********************
  parcels = list(range(rank,N_total,N_stride))

  # Generate spectrums for only my images
  spectra_set = [transmitted_info["spectra"].generate_recast_renormalized_image(image=idx,energy=7120.,total_flux=1e12) for idx in parcels]

  # Generate main sf
  spectra_main = transmitted_info["spectra"].generate_recast_renormalized_image(image=0,energy=7120.,total_flux=1e12)
  wavlen, flux, wavelength_A = next(spectra_main)
  direct_algo_res_limit = 1.7
  local_data = data()
  GF = gen_fmodel(resolution=direct_algo_res_limit,pdb_text=local_data.get("pdb_lines"),algorithm="fft",wavelength=wavelength_A)
  GF.set_k_sol(0.435)
  GF.make_P1_primitive()
  sfall_main = GF.get_amplitudes()
  
  # Generating sf for my wavelengths
  sfall_channels = []
  for x in range(len(wavlen)):
    GF.reset_wavelength(wavlen[x])
    GF.reset_specific_at_wavelength(
                     label_has="FE1",tables=local_data.get("Fe_oxidized_model"),newvalue=wavlen[x])
    GF.reset_specific_at_wavelength(
                     label_has="FE2",tables=local_data.get("Fe_reduced_model"),newvalue=wavlen[x])
    sfall_channels.append(GF.get_amplitudes())
    #print(wavlen[x], flux[x], sfall_channels[-1].size())

  # MU *********************
  
  #while len(parcels)>0:
  for i in range(len(parcels)):
    #import random
    #idx = random.choice(parcels)
    idx = parcels[i]
    cache_time = time()
    print("idx------start-------->",idx,"rank",rank,time())
    # if rank==0: os.system("nvidia-smi")
    tst_one(image=idx,spectra_iter=spectra_set[i],
            crystal=transmitted_info["crystal"],random_orientation=transmitted_info["random_orientations"][idx], sfall_channels=sfall_channels)
    #parcels.remove(idx)
    print("idx------finis-------->",idx,"rank",rank,time(),"elapsed",time()-cache_time)
  print("OK exiting rank",rank,"at",datetime.datetime.now(),"seconds elapsed",time()-start_elapse)
