from __future__ import division, print_function

from iotbx.phil import parse

help_message = '''
implement Max Likelihood refinement of the fp, fdp parameters.
'''

master_phil="""
N_total = 2000
    .type = int(value_min=0, value_max=100000)
    .help = How many elements of the total simulation to use for parameter refinement
    .help = Possible max working size for 64-core rack server is 6400
tester{
  rank = 45
    .type = int
    .help = For the purpose of testing a single worker rank, which rank number to simulate
  size = 512
    .type = int
    .help = For the purpose of testing a single worker rank, what total MPI size to simulate
}
starting_model{
  algorithm = to_file *from_file
  .type = choice
  .help = Compute the energy-dependent intensities and derivatives using chosen procedure
  .help = to_file choice computes the array in rank 0, then pickles it and exits the program
  .help = from_file reads intensities and static Fcalc from file
  filename = new_global_fdp_big_data.pickle
      .type = path
      .help = write out
  preset
    .help = Starting assumptions about the state of FE1 and FE2 scatterers
    .help = If FE1 is set to Fe_oxidized_model and FE2 is set to Fe_reduced_model then the
    .help = starting conditions of the parameter minimization are the same as computed in the
    .help = to_file reference.  In this case, the file reference intensities are used for iteration 1
    .help = and intensities are calculated on-the-fly for subsequent iterations.  Otherwise the
    .help = intensities are on-the-fly for all iterations.
  {
    FE1 = *Fe_oxidized_model Fe_reduced_model Fe_metallic_model
      .type = choice
    FE2 = Fe_oxidized_model *Fe_reduced_model Fe_metallic_model
      .type = choice
  }
}
LLG_evaluator{
  max_calls = 10
    .type = int
  enable_plot = False
    .type = bool
}
"""

phil_scope = parse(master_phil)
