import struct
import redis
import numpy as np

def toRedis(r, datas, label):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    s = datas.shape
   
    if len(s) == 1:
        shape = struct.pack('>II', s[0], 1)
    if len(s) == 2:
        shape = struct.pack('>II', s[0], s[1])
    if len(s) == 3:
        shape = struct.pack('>II', s[0], s[1], s[2])

    encoded = shape + datas.tobytes()

    # Store encoded data in Redis
    r.set(label, encoded)
    return

def fromRedis(r, label):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(label)

    if not encoded:
        return None

    s = struct.unpack('>II',encoded[:8])
    if s[1] == 1:
        s = (s[0], )

    a = np.frombuffer(encoded, dtype=np.float64, offset=8).reshape(s)
    return a