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
from dubbo.configs import ServiceConfig
from dubbo.proxy.handlers import RpcMethodHandler, RpcServiceHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate(request):
    print(f"Received request: {request.prompt}")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": request.prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512000
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    yield llm_pb2.GenerateReply(message=str(response))
    # yield llm_pb2.GenerateReply(message=str("test"))

if __name__ == "__main__":
    # build a method handler
    method_handler = RpcMethodHandler.server_stream(
        generate,
        request_deserializer=llm_pb2.GenerateRequest.FromString,
        response_serializer=llm_pb2.GenerateReply.SerializeToString,
    )
    # build a service handler
    service_handler = RpcServiceHandler(
        service_name="org.apache.dubbo.samples.proto.LlmService",
        method_handlers={"generate": method_handler},
    )

    service_config = ServiceConfig(service_handler)

    # start the server
    server = dubbo.Server(service_config).start()

    input("Press Enter to stop the server...\n")
