# distutils: sources = xxhash.c
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False
#include xxhash.h

cimport cython
import numpy as np
cimport numpy as np
import cython
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference, postincrement
from cython.parallel cimport prange


np.import_array()
ctypedef fused real:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.Py_hash_t
    cython.Py_UCS4

ctypedef fused real2:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t


cdef extern from "xxhash.h":
    ctypedef unsigned long long XXH64_hash_t
    cdef XXH64_hash_t XXH64(void* input, Py_ssize_t length, XXH64_hash_t seed) nogil


cpdef Py_ssize_t delete_horizontal_duplicates(
        real2[:, ::1] output_var_array,
        cython.uchar[:,::1] tmparray,
        cython.uchar max_reps_rows,
        Py_ssize_t tmparrayindexlen,
        Py_ssize_t width,
        Py_ssize_t[::1] tmparrayindex,


    ):
    cdef:
        Py_ssize_t i,number_index
        Py_ssize_t counter = 0
    with nogil:
        for i in range(tmparrayindexlen):
            for number_index in range(width):
                tmparray[i][output_var_array[i][number_index]]+=1 
                if tmparray[i][output_var_array[i][number_index]] > max_reps_rows:
                    break
            else:
                tmparrayindex[counter] = i
                counter = counter + 1
    return counter

cpdef void get_hash_from_row(
    real[:,:] argscopy_array, 
    Py_ssize_t[:] outi,
    Py_ssize_t lenargscopy_array,
    Py_ssize_t isize
    
):
    cdef:
        Py_ssize_t azinho
        Py_ssize_t lenori=argscopy_array.shape[1]
        Py_ssize_t buff =isize* lenori
        cython.uchar* ptr = NULL
        cdef real[:] a
    with nogil:
        for azinho in range(lenargscopy_array):
            a=argscopy_array[azinho]
            ptr = <cython.uchar*>(&a[0])
            outi[azinho]=XXH64(ptr,buff,1,)


cpdef Py_ssize_t deldummies(
    real[:, ::1] output_var_array,
    real[:, ::1] cleanedarray,
    real dummyvalid,
    Py_ssize_t y_axis,
    Py_ssize_t x_axis,

):
    cdef:
        Py_ssize_t cleanindex = 0
        Py_ssize_t i,j
    with nogil:
        for i in range(y_axis):
            for j in range(x_axis):
                if output_var_array[i][j] == dummyvalid:
                    break
            else:
                for j in range(x_axis):
                    cleanedarray[cleanindex][j] = output_var_array[i][j]
                cleanindex += 1
    return cleanindex


cpdef Py_ssize_t filter_vals(
    real[:, ::1] output_var_array,
    Py_ssize_t indexhashcol,
    Py_ssize_t[::1] hashcol,
    real[:, ::1] cleanedarray,
    ):
        cdef:
            unordered_map[Py_ssize_t, Py_ssize_t] resultdict
            unordered_map[Py_ssize_t, Py_ssize_t].iterator it
            Py_ssize_t h, widthloop
            Py_ssize_t resultcounter = 0
            Py_ssize_t width = output_var_array.shape[1]
        with nogil:
            for h in range(indexhashcol):
                resultdict[hashcol[h]] = h

            it=resultdict.begin()
            while(it != resultdict.end()):
                for widthloop in range(width):
                    cleanedarray[resultcounter][widthloop] = output_var_array[dereference(it).second][widthloop]
                postincrement(it)
                resultcounter+=1
            return resultcounter


cpdef void fastproduct(
    real2[::1] flat_index,
    real[:, ::1] output_var_array,
    real2[::1] lenlist,
    real[:,::1] empt,
    Py_ssize_t flat_indexllen_eshape,
    Py_ssize_t eshape,
    bint multicpu=True
    ):
    cdef:
        Py_ssize_t  n,o
    if multicpu:
        for o in prange(flat_indexllen_eshape,nogil=True):
            for n in range(eshape):
                output_var_array[..., n][o] = empt[n][(flat_index[o]) % (lenlist[n])]
                flat_index[o] = flat_index[o] // lenlist[n]
    else:
        for o in range(flat_indexllen_eshape):
            for n in range(eshape):
                output_var_array[..., n][o] = empt[n][(flat_index[o]) % (lenlist[n])]
                flat_index[o] = flat_index[o] // lenlist[n]        
