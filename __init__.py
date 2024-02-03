import numpy as np
from flatten_any_dict_iterable_or_whatsoever import set_in_original_iter
from copy import deepcopy

try:
    from .uniqueproductcython import (
        filter_vals,
        deldummies,
        fastproduct,
        get_hash_from_row,
        delete_horizontal_duplicates,
    )
except Exception:
    from cycompi import compile_cython_code
    import os

    numpyincludefolder = np.get_include()
    pyxfile = "uniqueproductcython.pyx"
    uniqueproductcythonmodule = pyxfile.split(".")[0]
    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    optionsdict = {
        "Options.docstrings": False,
        "Options.embed_pos_in_docstring": False,
        "Options.generate_cleanup_code": False,
        "Options.clear_to_none": True,
        "Options.annotate": True,
        "Options.fast_fail": False,
        "Options.warning_errors": False,
        "Options.error_on_unknown_names": True,
        "Options.error_on_uninitialized": True,
        "Options.convert_range": True,
        "Options.cache_builtins": True,
        "Options.gcc_branch_hints": True,
        "Options.lookup_module_cpdef": False,
        "Options.embed": False,
        "Options.cimport_from_pyx": False,
        "Options.buffer_max_dims": 8,
        "Options.closure_freelist_size": 8,
    }
    configdict = {
        "py_limited_api": False,
        "name": uniqueproductcythonmodule,
        "sources": [pyxfile_complete_path],
        "include_dirs": [numpyincludefolder],
        "define_macros": [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_USE_DICT_VERSIONS", 1),
            ("CYTHON_FAST_GIL", 1),
            ("CYTHON_USE_PYLIST_INTERNALS", 1),
            ("CYTHON_USE_UNICODE_INTERNALS", 1),
            ("CYTHON_ASSUME_SAFE_MACROS", 1),
            ("CYTHON_USE_TYPE_SLOTS", 1),
            ("CYTHON_USE_PYTYPE_LOOKUP", 1),
            ("CYTHON_USE_ASYNC_SLOTS", 1),
            ("CYTHON_USE_PYLONG_INTERNALS", 1),
            ("CYTHON_USE_UNICODE_WRITER", 1),
            ("CYTHON_UNPACK_METHODS", 1),
            ("CYTHON_USE_EXC_INFO_STACK", 1),
            ("CYTHON_ATOMICS", 1),
        ],
        "undef_macros": [],
        "library_dirs": [],
        "libraries": [],
        "runtime_library_dirs": [],
        "extra_objects": [],
        "extra_compile_args": ["/O2", "/Oy", "/std:c++20", "/openmp"],
        "extra_link_args": [],
        "export_symbols": [],
        "swig_opts": [],
        "depends": [],
        "language": "c++",
        "optional": None,
    }
    compiler_directives = {
        "binding": True,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "overflowcheck": False,
        "overflowcheck.fold": False,
        "embedsignature": False,
        "embedsignature.format": "c",  # (c / python / clinic)
        "cdivision": True,
        "cdivision_warnings": False,
        "cpow": True,
        "always_allow_keywords": False,
        "c_api_binop_methods": False,
        "profile": False,
        "linetrace": False,
        "infer_types": True,
        "language_level": "3str",  # (2/3/3str)
        "c_string_type": "bytes",  # (bytes / str / unicode)
        "c_string_encoding": "ascii",  # (ascii, default, utf-8, etc.)
        "type_version_tag": False,
        "unraisable_tracebacks": True,
        "iterable_coroutine": False,
        "annotation_typing": False,
        "emit_code_comments": False,
        "cpp_locals": False,
        "legacy_implicit_noexcept": False,
        "optimize.use_switch": True,
        "optimize.unpack_method_calls": True,
        "warn.undeclared": False,  # (default False)
        "warn.unreachable": True,  # (default True)
        "warn.maybe_uninitialized": False,  # (default False)
        "warn.unused": False,  # (default False)
        "warn.unused_arg": False,  # (default False)
        "warn.unused_result": False,  # (default False)
        "warn.multiple_declarators": True,  # (default True)
        "show_performance_hints": True,  # (default True)
    }

    compile_cython_code(
        name=uniqueproductcythonmodule,
        configdict=configdict,
        optionsdict=optionsdict,
        cmd_line_args=compiler_directives,
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    from  .uniqueproductcython import (
        filter_vals,
        deldummies,
        fastproduct,
        get_hash_from_row,
        delete_horizontal_duplicates,
    )


def hash_2d_nparray(arr):
    if not arr.flags["C_CONTIGUOUS"] or not arr.flags["ALIGNED"]:
        arr = arr.copy()
    lenargscopy_array = len(arr)
    isize = arr.itemsize
    hashcol = np.zeros(arr.shape[0], dtype=np.int64)
    get_hash_from_row(arr, hashcol, lenargscopy_array, isize)
    return hashcol


def _generate_product_np(
    argscopy_array,
    remove_duplicates=True,
    str_format_function=repr,
    multicpu=True,
    return_index_only=False,
    max_reps_rows=-1,
    r=-1,
    dummyval="DUMMYVAL",
):
    lenli = [argscopy_array.shape[1] for x in range(len(argscopy_array))]
    output_array_len = np.prod(lenli)
    output_var_array = np.zeros(
        (output_array_len, len(argscopy_array)), dtype=argscopy_array.dtype
    )
    flat_index = np.arange(output_array_len * len(argscopy_array), dtype=np.int64)
    flat_indexllen = len(flat_index)
    empt = np.ascontiguousarray(argscopy_array)
    eshape = output_var_array.shape[1]

    lenlist = np.array(lenli, dtype=np.int64)
    flat_indexllen_eshape = flat_indexllen // eshape
    fastproduct(
        flat_index,
        output_var_array,
        lenlist,
        empt,
        flat_indexllen_eshape,
        eshape,
        multicpu,
    )
    if remove_duplicates:
        lenargscopy_array = len(output_var_array)
        isize = output_var_array.itemsize
        hashcol = np.zeros(output_var_array.shape[0], dtype=np.int64)
        get_hash_from_row(output_var_array, hashcol, lenargscopy_array, isize)
        indexhashcol = len(hashcol)
        cleanedarray = np.empty_like(output_var_array)
        resultcounter = filter_vals(
            output_var_array,
            indexhashcol,
            hashcol,
            cleanedarray,
        )
        output_var_array = cleanedarray[:resultcounter]
    return output_var_array


def create_window_list(args1, r=0):
    windowlist = []
    if isinstance(args1, np.ndarray):
        args1 = args1.tolist()
    for arg in args1:
        lenarg = len(arg)
        if r:
            lenarg = r if len(arg) > r else lenarg
        for index in range(lenarg):
            windowlist.append([])
            windowlist[-1].extend(arg[index:])
            windowlist[-1].extend(arg[:index])
    return windowlist


def product_with_r(
    args,
    r,
):
    a = create_window_list(args, r)
    if isinstance(args, np.ndarray):
        a = np.array(a).astype(args.dtype)
    return a


def generate_product(
    args,
    remove_duplicates=True,
    str_format_function=repr,
    multicpu=True,
    return_index_only=False,
    max_reps_rows=-1,
    r=-1,
    dummyval="DUMMYVAL",
    debug=False,
    try_always_numpy=True,
):
    isnp = isinstance(args, np.ndarray)
    if isnp and (return_index_only or max_reps_rows > 1 or r > 1):
        args = args.tolist()
        try_always_numpy = False
    if r > 1:
        args = product_with_r(
            args=args,
            r=r,
        )
    isnp = isinstance(args, np.ndarray)
    argsnumpytest = np.array([], dtype=object)
    if (isnp and args.dtype != object) or try_always_numpy:
        if try_always_numpy:
            if (
                not isinstance(args, np.ndarray)
                and set([len(argsh) for argsh in args]) == 1
            ):
                try:
                    argsnumpytest = np.asanyarray(args)
                except Exception as e:
                    pass

        try:
            return _generate_product_np(
                argscopy_array=args,
                r=r,
                remove_duplicates=remove_duplicates,
                dummyval=dummyval,
                str_format_function=str_format_function,
                multicpu=multicpu,
                return_index_only=return_index_only,
                max_reps_rows=-1,
            )
        except Exception as e:
            pass
    if argsnumpytest.dtype != object:
        try:
            return _generate_product_np(
                argscopy_array=args,
                r=r,
                remove_duplicates=remove_duplicates,
                dummyval=dummyval,
                str_format_function=str_format_function,
                multicpu=multicpu,
                return_index_only=return_index_only,
                max_reps_rows=-1,
            )
        except Exception as e:
            pass
    argscpy = deepcopy(args)
    specialdict = {}
    for rl1 in range(len(argscpy)):
        suli = argscpy[rl1]
        for rl2 in range(len(suli)):
            v = suli[rl2]
            try:
                hash(v)
            except Exception:
                v = str_format_function(v)
                args[rl1][rl2] = v
                specialdict[v] = suli[rl2]
    original_lookup_list_passed = []
    longest_array = 0
    smallest_array = 9999999
    for x in args:
        if longest_array < len(x):
            longest_array = len(x)
        if smallest_array > len(x):
            smallest_array = len(x)
        for y in x:
            original_lookup_list_passed.append(((str_format_function(y)), y))
    original_lookup_dict_passed = dict(original_lookup_list_passed)
    all_lists = []
    for ax in args:
        if smallest_array < longest_array:
            dummylist = [dummyval] * (longest_array - len(ax))
            all_lists.append(dummylist + ax)
        else:
            all_lists.append(ax)
    allarras = [longest_array for x in range(len(args))]
    argscopy = deepcopy(all_lists)
    flattend = []
    for rl1 in range(len(argscopy)):
        suli = argscopy[rl1]
        for rl2 in range(len(suli)):
            v = suli[rl2]
            try:
                hash(v)
            except Exception:
                v = str_format_function(v)
            flattend.append((v, (rl1, rl2)))

    repruniqueid = {}
    repruniqueidv = {}
    repruniquerepr = {}
    ordered_in_one_list_id = []
    ordered_in_one_list_original = []
    idcounter = 0

    for repri, keys in flattend:
        try:
            if repri not in repruniquerepr:
                repruniquerepr[repri] = original_lookup_dict_passed.get(
                    str_format_function(repri)
                )
                repruniqueid[repri] = idcounter
                repruniqueidv[idcounter] = repri
                ordered_in_one_list_id.append(idcounter)
                ordered_in_one_list_original.append(repruniquerepr[repri])
                try:
                    ordered_in_one_list_original[-1] = specialdict[repri]
                except Exception as e:
                    pass

                idcounter += 1
        except Exception as e:
            repri = str(repri)
            repruniquerepr[repri] = original_lookup_dict_passed.get(
                str_format_function(repri)
            )
            repruniqueid[repri] = idcounter
            repruniqueidv[idcounter] = repri
            ordered_in_one_list_id.append(idcounter)
            ordered_in_one_list_original.append(repruniquerepr[repri])
            idcounter += 1

        try:
            set_in_original_iter(
                iterable=argscopy, keys=keys, value=repruniqueid[repri]
            )
        except Exception as e:
            pass
    if idcounter < 256:
        numpydtype = np.uint8
    elif idcounter < 65536:
        numpydtype = np.uint16

    elif idcounter < 4294967296:
        numpydtype = np.uint32
    else:
        numpydtype = np.int64
    if debug:
        print(f"{original_lookup_list_passed=}")
        print(f"{str_format_function=}")
        print(f"{idcounter=}")
        print(f"{args=}")
        print(f"{original_lookup_dict_passed=}")
        print(f"{all_lists=}")
        print(f"{smallest_array=}")
        print(f"{longest_array=}")
        print(f"{flattend=}")
        print(f"{argscopy=}")
        print(f"{allarras=}")
        print(f"{repruniqueid=}")
        print(f"{repruniqueidv=}")
        print(f"{repruniquerepr=}")
        print(f"{numpydtype=}")
        print(f"{repruniquerepr=}")
        print(f"{numpydtype=}")
        print(f"{ordered_in_one_list_id=}")
        print(f"{ordered_in_one_list_original=}")

    argscopy_array = np.array(argscopy, dtype=numpydtype)
    output_array_len = np.prod(allarras)
    output_var_array = np.zeros((output_array_len, len(allarras)), dtype=np.int64)
    flat_index = np.arange(output_array_len * len(allarras), dtype=np.int64)
    flat_indexllen = len(flat_index)
    empt = argscopy_array.copy().astype(np.int64)
    eshape = output_var_array.shape[1]
    lenlist = np.array(allarras, dtype=np.int64)
    flat_indexllen_eshape = flat_indexllen // eshape
    fastproduct(
        flat_index,
        output_var_array,
        lenlist,
        empt,
        flat_indexllen_eshape,
        eshape,
        multicpu,
    )

    if dummyval in repruniqueid:
        dummyvalid = repruniqueid.get(dummyval)
        cleanedarray = np.zeros_like(output_var_array)
        y_axis = output_var_array.shape[0]
        x_axis = output_var_array.shape[1]
        cleanindex = deldummies(
            output_var_array,
            cleanedarray,
            dummyvalid,
            y_axis,
            x_axis,
        )
        output_var_array = cleanedarray[:cleanindex]
    if remove_duplicates:
        lenargscopy_array = len(output_var_array)
        isize = output_var_array.itemsize

        hashcol = np.zeros(output_var_array.shape[0], dtype=np.int64)
        get_hash_from_row(output_var_array, hashcol, lenargscopy_array, isize)

        indexhashcol = len(hashcol)
        cleanedarray = np.zeros_like(output_var_array)
        resultcounter = filter_vals(
            output_var_array,
            indexhashcol,
            hashcol,
            cleanedarray,
        )
        output_var_array = cleanedarray[:resultcounter]

    if max_reps_rows > 0:
        if max_reps_rows > 255:
            max_reps_rows = 255
        tmparray = np.zeros((output_var_array.shape[0], idcounter), dtype=np.uint8)
        tmparrayindex = np.zeros(output_var_array.shape[0], dtype=np.int64)
        tmparrayindexlen = tmparrayindex.shape[0]
        width = output_var_array.shape[1]
        counter = delete_horizontal_duplicates(
            output_var_array,
            tmparray,
            max_reps_rows,
            tmparrayindexlen,
            width,
            tmparrayindex,
        )

        output_var_array = output_var_array[tmparrayindex[:counter]]

    if return_index_only:
        return output_var_array
    try:
        original_valsarray = np.asanyarray(ordered_in_one_list_original)
    except Exception:
        original_valsarray = np.asanyarray(ordered_in_one_list_original, dtype="object")
    return original_valsarray[output_var_array.ravel()].reshape(output_var_array.shape)
