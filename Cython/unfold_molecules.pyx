cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def unravel_list(list my_list):
    cdef int total_elements = 0
    cdef list result = []

    for sublist in my_list:
        total_elements += len(sublist)

    result = [0] * total_elements
    cdef int index = 0

    for sublist in my_list:
        cdef int sublist_length = len(sublist)
        cdef int i
        for i in range(sublist_length):
            result[index] = sublist[i]
            index += 1

    return result