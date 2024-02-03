# Efficient NumPy Permutations, Combinations, and Product using Cython/C++/OpenMP


Generate permutations, combinations, and product sets with NumPy efficiently. 
The provided `generate_product` function is designed to outperform the standard 
itertools library, offering more than 20x speed improvement.




- Utilizes a "Yellow-line-free" Cython Backend for high speed performance.
- Implements OpenMP multiprocessing for parallel processing.
- Compiles on the first run (requires a C/C++ compiler installed on your PC).
- Achieves 90% less memory usage compared to itertools.
- Performance scales with data size, making it ideal for large datasets.
- Efficiently creates a lookup NumPy array with a lightweight dtype (typically np.uint8, unless you are combining more than 255 different elements).
- Utilizes numpy indexing for memory savings - depending on the datatype (and your luck :-) ), numpy shows you only element views, which means, you are saving a loooooooooooooooooooot of memory

## Supported Functionality


<table><thead><tr><th><p>Iterator</p></th><th><p>Arguments</p></th><th><p>Results</p></th></tr></thead><tbody><tr><td><p><a href="https://docs.python.org/3/library/itertools.html#itertools.product" title="itertools.product"><code><span>product()</span></code></a></p></td><td><p>p, q, … [repeat=1]</p></td><td><p>cartesian product, equivalent to a nested for-loop</p></td></tr><tr><td><p><a href="https://docs.python.org/3/library/itertools.html#itertools.permutations" title="itertools.permutations"><code><span>permutations()</span></code></a></p></td><td><p>p[, r]</p></td><td><p>r-length tuples, all possible orderings, no repeated elements</p></td></tr><tr><td><p><a href="https://docs.python.org/3/library/itertools.html#itertools.combinations" title="itertools.combinations"><code><span>combinations()</span></code></a></p></td><td><p>p, r</p></td><td><p>r-length tuples, in sorted order, no repeated elements</p></td></tr><tr><td><p><a href="https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement" title="itertools.combinations_with_replacement"><code><span>combinations_with_replacement()</span></code></a></p></td><td><p>p, r</p></td><td><p>r-length tuples, in sorted order, with repeated elements</p></td></tr></tbody></table>


## Getting Started

### Only tested on Windows 10 / Python 3.11

```python
- Make sure you have Python and a C/C++ compiler installed 
- Use pip install numpycythonpermutations or download it from Github
```


## Some examples

## Generating all RGB colors in 200 ms.

#### more than 25 times faster than itertools generating all RGB colors


```python
import numpy as np
import itertools
from numpycythonpermutations import generate_product

# RGB COLORS:

args = np.asarray( # The input must be always 2 dimensional (list or numpy)
    [
        list(range(256)),
        list(range(256)),
        list(range(256)),
    ],
    dtype=np.uint8,
)

In [17]: %timeit resus = np.array(list(itertools.product(*args)))
5.88 s ± 78.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

...: %timeit resus = generate_product(args, remove_duplicates=False, str_format_function=repr, multicpu=True, return_index_only=False, max_reps_rows=-1, r=-1, dummyval="DUMMYVAL")
232 ms ± 31.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

```

### But even 2.5x faster when using a tiny database 


```python
2.5x times faster using little data
args = np.asarray(
    [
        list(range(5)),
        list(range(5)),
        list(range(5)),
    ],
    dtype=np.uint8,
)

In [23]: %timeit np.array(list(itertools.product(*args)))
39.3 µs ± 113 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

In [25]: %timeit resus = generate_product(args, remove_duplicates=False, str_format_function=repr, multicpu=True, return_index_only=False, max_reps_rows=-1, r=-1, dummyval="DUMMYVAL")
19.2 µs ± 176 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
```

### Attention! The output is different (Fortran-styled order) from itertools:

#### Itertools

```python


array([
[  0,   0,   0],
[  1,   0,   0],
[  2,   0,   0],
...,
[253, 255, 255],
[254, 255, 255],
[255, 255, 255]], dtype=np.uint8)

```

#### numpycythonpermutations


```python
array(
     [[  0,   0,   0],
      [  0,   0,   1],
      [  0,   0,   2],
      ...,
      [255, 255, 253],
      [255, 255, 254],
      [255, 255, 255]])
```

## Deleting duplicates      

```python
args = [
    [1, 2, 3, 4],
    [2, 0, 0, 2],
    [2, 1, 6, 2],
    [1, 2, 3, 4],
]
resus1 = generate_product(
    args,
    remove_duplicates=True,
)

print(resus1)
print(resus1.shape)
In [15]: resus1
Out[15]:
array([[1, 2, 2, 1],
[2, 0, 2, 3],
[2, 2, 2, 1],
[3, 2, 2, 1],
...
[4, 0, 1, 4],
[1, 2, 6, 4],
[3, 2, 6, 4],
[4, 2, 6, 4],
[1, 0, 6, 4],
[3, 0, 6, 4],
[4, 0, 6, 4]])
In [18]: resus1.shape
Out[18]: (96, 4)

# Without removing duplicates

args = [
    [1, 2, 3, 4],
    [2, 0, 0, 2],
    [2, 1, 6, 2],
    [1, 2, 3, 4],
]
resus2 = generate_product(
    args,
    remove_duplicates=False,
)
print(resus2.shape)

In [16]: resus2
Out[16]:
array([[1, 2, 2, 1],
[2, 2, 2, 1],
[3, 2, 2, 1],
...,
[2, 2, 2, 4],
[3, 2, 2, 4],
[4, 2, 2, 4]])
In [17]: resus2.shape
Out[17]: (256, 4)
```

## Filtering Data

### To get all colors whose RGB values are R!=G!=B

#### The order of any filtered output may vary each time due to multicore parsing.

```python

args = [
    list(range(256)),
    list(range(256)),
    list(range(256)),
]

generate_product(args, max_reps_rows=1)

array([[119, 158, 238],
[ 50,   2,   0],
[226, 251,  90],
...,
[244, 254, 255],
[245, 254, 255],
[246, 254, 255]])

# But it takes some time to filter 16,7 Million colors:

In [38]: %timeit generate_product(args, max_reps_rows=1)
11.7 s ± 437 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Passing a NumPy array is a little faster

args = np.asarray(
    [
        list(range(256)),
        list(range(256)),
        list(range(256)),
    ],
    dtype=np.uint8,
)

In [2]: %timeit generate_product(args, max_reps_rows=1)
9.94 s ± 209 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Another example
args = [
    [2, 1, 3, 4],
    [4, 4, 3, 4],
]
resus = generate_product(args, 
remove_duplicates=True, # removes all duplicated rows
r=len(args[0]), # similar to itertools
max_reps_rows=2) # allows only 2 occurrences of the same element in the same row

[[1 1 2 2 4 3 3 4]
[1 1 2 2 4 3 4 3]
[1 2 2 1 3 4 3 4]
[2 1 2 1 3 4 4 3]
[1 1 2 2 3 3 4 4]
[2 1 1 2 3 4 3 4]
[2 1 2 1 3 4 3 4]
[1 1 2 2 3 4 3 4]


# Another example

args = [
    [1, 2, 3, 4],
]

resus = generate_product(args, remove_duplicates=False, r=len(args[0]))
print(resus)
print(resus.shape)

[[1 2 3 4]
[2 2 3 4]
[3 2 3 4]
...
[2 1 2 3]
[3 1 2 3]
[4 1 2 3]]
(256, 4)

```

## You can mix data types


```python
args = [
    [[1, 2], 3, 4],
    [3, "xxxxx", 3, 6],
    [2, 0, 0, 2],
    [2, 0, [0, 2]],
    [8, 2, 8, 2],
    [4, 5, 4, 5],
    [[3, 3], 3, 6],
    [4, 5, 4, 5],
    [0, {2, 3, 4}, 8, 7],
    [1, 2, b"xxx3", 4],
]

q = generate_product(args, remove_duplicates=False)

Out[6]:
array([[list([1, 2]), 3, 2, ..., 4, 0, 1],
[3, 3, 2, ..., 4, 0, 1],
[4, 3, 2, ..., 4, 0, 1],
...,
[list([1, 2]), 6, 2, ..., 5, 7, 4],
[3, 6, 2, ..., 5, 7, 4],
[4, 6, 2, ..., 5, 7, 4]], dtype=object)



the function repr is usually used to filter Not-Numpy-Friendly-Data
This might lead to some problems, e.g. pandas DataFrames which are usually not
fully shown when calling __repr__
In these cases, you can pass a custom function to str_format_function
 (but to be honest: Who the hell puts a pandas DataFrame inside a NumPy array?)

# Example for a function (The string is only used for indexing)
str_format_function = (
    lambda x: x.to_string() if isinstance(x, pd.DataFrame) else repr(x)
)


import pandas as pd

args = [
    [2, 1, 3, 4],
    [4, 4, 3, 4],
    [
        pd.read_csv(
            "https://github.com/datasciencedojo/datasets/blob/master/titanic.csv",
            on_bad_lines="skip",
        ),
        np.array([222, 3]),
        dict(baba=333, bibi=444),
    ],
]

resus = generate_product(
    args,
    remove_duplicates=True,
    r=len(args[0]),
    max_reps_rows=-1,
    str_format_function=str_format_function,
)
print(resus)
print(resus.shape)

Ain't it pretty? hahaha

[[4 3 2 ... {'baba': 333, 'bibi': 444}
<!DOCTYPE html>
0                                                 <html
1                                             lang="en"
2       data-color-mode="auto" data-light-theme="lig...
3       data-a11y-animated-images="system" data-a11y...
4                                                     >
...                                                 ...
1062                                             </div>
1063      <div id="js-global-screen-reader-notice" c...
1064      <div id="js-global-screen-reader-notice-as...
1065                                            </body>
1066                                            </html>

[1067 rows x 1 columns]
array([222,   3])]
[2 1 1 ... {'baba': 333, 'bibi': 444} {'baba': 333, 'bibi': 444}
array([222,   3])]
[1 1 3 ...                                         <!DOCTYPE html>
0                                                 <html
1                                             lang="en"
2       data-color-mode="auto" data-light-theme="lig...
3       data-a11y-animated-images="system" data-a11y...
4                                                     >
...                                                 ...
1062                                             </div>

```

## Inhomogeneous Shapes? No problem!

```python 
# An Inhomogeneous Shape is also no problem. 
# Just make sure that the default dummy value dummyval="DUMMYVAL" is not in your Array (not very likely, I guess)

a = [1, 2]
b = [3, 4]
c = [5, 6, 7]
d = [8, 9, 10]
total = [a, b, c, d]

resus = generate_product(total, remove_duplicates=True, dummyval="DUMMYVAL")
print(resus)

[[2 3 6 9]
[1 3 5 8]
[1 3 6 9]
[1 4 6 8]
[1 4 5 8]
[2 3 5 8]
[1 4 7 9]
[1 3 7 9]
...
[2 3 7 9]
[2 4 7 9]
[2 3 5 10]
[2 4 5 10]
[1 3 6 10]
[2 3 6 10]
[1 4 6 10]
[2 4 6 10]
[2 4 7 10]]

a = [1, 2, 3]
b = [3, 4, 4]
c = [5, 6]
d = [8, 9, 10]
total = [a, b, c, d]

resus = generate_product(total, remove_duplicates=True, dummyval="DUMMYVAL")
print(resus)
[[1 3 5 8]
[3 4 6 10]
[1 3 5 10]
[2 4 5 9]
[2 4 6 8]
[3 3 5 8]
...
[3 3 6 9]
[2 3 6 10]
[2 4 6 9]
[1 4 6 10]
[3 4 6 9]
[3 3 5 10]
[1 3 6 10]
[1 4 5 10]
[2 4 5 10]
[3 4 5 10]
[3 3 6 10]]

``` 


## How to get the index

```python 

# To save memory, the function can only return the index, this saves a lot of memory 
# and you can access each element by looping through the data and accessing the input Element

args = [
    [100, 200, 300, 400],
    [300, 300, 300, 600],
    [200, 000, 000, 200],
    [200, 000, 000, 200],
    [800, 200, 800, 200],
    [400, 500, 400, 500],
    [300, 300, 300, 600],
    [400, 500, 400, 500],
    [000, 900, 800, 700],
    [100, 200, 300, 400],
]

resus = generate_product(
    args,
    remove_duplicates=False,
    return_index_only=True,
)
print(resus)
print(resus.shape)

The function returns:
[[0 2 1 ... 3 5 0]
[1 2 1 ... 3 5 0]
[2 2 1 ... 3 5 0]
...
[1 4 1 ... 7 9 3]
[2 4 1 ... 7 9 3]
[3 4 1 ... 7 9 3]]
(1048576, 10)
