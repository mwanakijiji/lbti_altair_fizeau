# This is for checking the number of CPU across multiple nodes

# Created 2020 Feb. 28 by E.S.

import multiprocessing

ncpu = multiprocessing.cpu_count()

print("Number of CPU:")
print(ncpu)
