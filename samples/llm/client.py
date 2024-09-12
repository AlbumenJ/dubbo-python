#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from proto import llm_pb2

import dubbo
from dubbo.configs import ReferenceConfig


class GreeterServiceStub:
    def __init__(self, client: dubbo.Client):
        self.generate = client.server_stream(
            method_name="generate",
            request_serializer=llm_pb2.GenerateRequest.SerializeToString,
            response_deserializer=llm_pb2.GenerateReply.FromString,
        )

    def generate(self, values):
        return self.generate(values)


if __name__ == "__main__":
    reference_config = ReferenceConfig.from_url(
        "tri://127.0.0.1:50051/org.apache.dubbo.samples.proto.LlmService?timeout=10000000"
    )
    dubbo_client = dubbo.Client(reference_config)

    stub = GreeterServiceStub(dubbo_client)

    result = stub.generate(llm_pb2.GenerateRequest(prompt=input("Please input your question: ")))

    print(str(result))
