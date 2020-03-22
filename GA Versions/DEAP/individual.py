import struct
from _codecs import decode

import numpy as np
from bitarray import bitarray


class individual:
    def __init__(self, weights):
        self.weights = weights
        self.vector_weights = self.flatten(self.weights)
        self.binary_vector_weights = self.nparray_to_binarray(self.vector_weights)

    def set_mat_from_vect(self):
        self.weights = self.unflatten(self.vector_weights, self.weights)

    def set_vec_from_mat(self):
        self.vector_weights = self.flatten(self.weights)

    def _flatten(self, values):
        if isinstance(values, np.ndarray) and not isinstance(values[0], np.ndarray):
            yield values.flatten()
        else:
            for value in values:
                yield from self._flatten(value)

    def flatten(self, values):
        # flatten nested lists of np.ndarray to np.ndarray
        return np.concatenate(list(self._flatten(values)))

    def _unflatten(self, flat_values, prototype, offset):
        if isinstance(prototype, np.ndarray) and not isinstance(prototype[0], np.ndarray):
            shape = prototype.shape
            new_offset = offset + np.product(shape)
            value = flat_values[offset:new_offset].reshape(shape)
            return value, new_offset
        else:
            result = []
            for value in prototype:
                value, offset = self._unflatten(flat_values, value, offset)
                result.append(value)
            return np.asarray(result), offset

    def unflatten(self, flat_values, prototype):
        # unflatten np.ndarray to nested lists with structure of prototype
        result, offset = self._unflatten(flat_values, prototype, 0)
        assert (offset == len(flat_values))
        return result

    def nparray_to_binarray(self, a):
        def float_to_bin(value):  # For testing.
            """ Convert float to 64-bit binary string. """
            [d] = struct.unpack(">Q", struct.pack(">d", value))
            return '{:064b}'.format(d)

        a = np.array(list(map(lambda x: float_to_bin(x), a)))
        binarr = ''
        for x in a:
            binarr += x
        binarr = bitarray(binarr)

        return binarr

    def binarray_to_nparray(self, binarr, arr_shape):
        def bin_to_float(b):
            """ Convert binary string to a float. """
            bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
            return struct.unpack('>d', bf)[0]

        def int_to_bytes(n, length):  # Helper function
            """ Int/long to byte string.

                Python 3.2+ has a built-in int.to_bytes() method that could be used
                instead, but the following works in earlier versions including 2.x.
            """
            return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

        # print(arr_shape)
        b = np.zeros(arr_shape)
        bidx = 0
        for idx in range(0, len(binarr), 64):
            # print(bin_to_float(binarr[idx:idx + 64].to01()))
            b[bidx] = bin_to_float(binarr[idx:idx + 64].to01())
            bidx += 1
        return b