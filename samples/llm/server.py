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

    # 使用 tokenizer 将消息模板转换为文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 将输入文本转换为模型可以接受的张量，并确保 input_ids 类型为 LongTensor
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    model_inputs.input_ids = model_inputs.input_ids.long()

    # 逐步生成 token 并实时返回
    generated_ids = []
    stop_token = "<|endoftext|>"  # 假设这是结束标记

    # 设置每次生成的 token 数量和循环生成
    for step in range(0, 512000, 50):
        output = model.generate(
            model_inputs.input_ids,
            max_new_tokens=50,  # 每次生成少量 token
            output_scores=False,
            return_dict_in_generate=True,
            do_sample=False  # 确保生成是确定性的
        )

        # 提取生成的新 token
        new_tokens = output.sequences[:, model_inputs.input_ids.shape[-1]:]
        generated_ids.append(new_tokens)

        # 解码新生成的 token 为文本
        partial_response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        # 通过 yield 返回部分生成结果
        yield llm_pb2.GenerateReply(message=str(partial_response))

        # 如果生成的内容包含结束标记，提前停止生成
        if stop_token in partial_response:
            break

        # 将新生成的文本加入到下次输入中，确保模型使用完整的上下文
        model_inputs = tokenizer(
            text + partial_response,  # 在原始输入的基础上加上新生成的部分
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).to("cuda")
        model_inputs.input_ids = model_inputs.input_ids.long()  # 确保类型为 LongTensor


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
