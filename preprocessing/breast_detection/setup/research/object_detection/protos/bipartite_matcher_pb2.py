# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/bipartite_matcher.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n/object_detection/protos/bipartite_matcher.proto\x12\x17object_detection.protos\"4\n\x10\x42ipartiteMatcher\x12 \n\x11use_matmul_gather\x18\x06 \x01(\x08:\x05\x66\x61lse')



_BIPARTITEMATCHER = DESCRIPTOR.message_types_by_name['BipartiteMatcher']
BipartiteMatcher = _reflection.GeneratedProtocolMessageType('BipartiteMatcher', (_message.Message,), {
  'DESCRIPTOR' : _BIPARTITEMATCHER,
  '__module__' : 'object_detection.protos.bipartite_matcher_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.BipartiteMatcher)
  })
_sym_db.RegisterMessage(BipartiteMatcher)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BIPARTITEMATCHER._serialized_start=76
  _BIPARTITEMATCHER._serialized_end=128
# @@protoc_insertion_point(module_scope)
