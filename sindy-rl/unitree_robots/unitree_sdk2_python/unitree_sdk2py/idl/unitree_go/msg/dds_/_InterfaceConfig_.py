"""
  Generated by Eclipse Cyclone DDS idlc Python Backend
  Cyclone DDS IDL version: v0.11.0
  Module: unitree_go.msg.dds_
  IDL file: InterfaceConfig_.idl

"""

from enum import auto
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
# import unitree_go


@dataclass
@annotate.final
@annotate.autoid("sequential")
class InterfaceConfig_(idl.IdlStruct, typename="unitree_go.msg.dds_.InterfaceConfig_"):
    mode: types.uint8
    value: types.uint8
    reserve: types.array[types.uint8, 2]


