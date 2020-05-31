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
    worker_action_datas = ListField()

class Weight(Document):
    weight  = FloatField()
    dim1    = IntField()
    dim2    = IntField()
    dim3    = IntField() 