# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: llm.proto
# Protobuf Python Version: 5.27.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    1,
    '',
    'llm.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tllm.proto\x12\x1eorg.apache.dubbo.samples.proto\"!\n\x0fGenerateRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\" \n\rGenerateReply\x12\x0f\n\x07message\x18\x01 \x01(\t2z\n\nLlmService\x12l\n\x08generate\x12/.org.apache.dubbo.samples.proto.GenerateRequest\x1a-.org.apache.dubbo.samples.proto.GenerateReply0\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'llm_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_GENERATEREQUEST']._serialized_start=45
  _globals['_GENERATEREQUEST']._serialized_end=78
  _globals['_GENERATEREPLY']._serialized_start=80
  _globals['_GENERATEREPLY']._serialized_end=112
  _globals['_LLMSERVICE']._serialized_start=114
  _globals['_LLMSERVICE']._serialized_end=236
# @@protoc_insertion_point(module_scope)
