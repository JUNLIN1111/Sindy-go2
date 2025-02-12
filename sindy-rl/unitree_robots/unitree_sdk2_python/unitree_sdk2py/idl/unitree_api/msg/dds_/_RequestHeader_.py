"""
  Generated by Eclipse Cyclone DDS idlc Python Backend
  Cyclone DDS IDL version: v0.10.2
  Module: unitree_api.msg.dds_
  IDL file: RequestHeader_.idl

"""

from enum import auto
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
# import unitree_api


@dataclass
@annotate.final
@annotate.autoid("sequential")
class RequestHeader_(idl.IdlStruct, typename="unitree_api.msg.dds_.RequestHeader_"):
    identity: 'unitree_sdk2py.idl.unitree_api.msg.dds_.RequestIdentity_'
    lease: 'unitree_sdk2py.idl.unitree_api.msg.dds_.RequestLease_'
    policy: 'unitree_sdk2py.idl.unitree_api.msg.dds_.RequestPolicy_'


