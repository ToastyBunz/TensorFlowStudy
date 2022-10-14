import tensorflow as tf
import numpy as np



a = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
     [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
     [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

for item in ds:
    print(ds)

for item in ds.as_numpy_iterator():
    print(item)