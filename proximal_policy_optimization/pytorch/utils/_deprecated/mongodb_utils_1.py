import struct
import numpy as np
from pymongo import MongoClient
from mongoengine import *

class Observation(Document):
    states              = ListField()
    actions             = ListField()    
    rewards             = FloatField()
    dones               = FloatField()     
    next_states         = ListField()
    logprobs            = ListField()
    next_next_states    = ListField()

class Weight(Document):
    weight  = FloatField()
    dim1    = IntField()
    dim2    = IntField()
    dim3    = IntField() 