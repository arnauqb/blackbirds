try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
except:
    mpi_comm = None
    mpi_rank = 0
    mpi_size = 1
