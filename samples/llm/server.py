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
from dubbo.configs import ServiceConfig, RegistryConfig
from dubbo.proxy.handlers import RpcMethodHandler, RpcServiceHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

def generate(request):
    print(f"Received request: {request.prompt}")

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
    attention_mask = model_inputs['attention_mask']

    # 逐步生成 token 并实时返回
    generated_ids = []
    max_steps = 100  # 最大生成步数，防止无限循环
    for step in range(max_steps):
        # 生成文本
        output = model.generate(
            model_inputs.input_ids,
            max_new_tokens=20,  # 每次生成 20 个 token
            attention_mask=attention_mask,  # 传递 attention_mask
            output_scores=False,
            return_dict_in_generate=True,
            do_sample=True,  # 开启采样生成
            temperature=0.7,  # 控制生成文本的多样性
            top_p=0.8,  # 核采样
            top_k=20  # top-k 采样
        )

        # 提取生成的新 token
        new_tokens = output.sequences[:, model_inputs.input_ids.shape[-1]:]
        generated_ids.append(new_tokens)

        # 解码新生成的 token 为文本
        partial_response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        # 通过 yield 返回部分生成结果
        print(f"Step {step}: {partial_response}")

        # 如果新生成的 token 为空，则停止生成
        if not partial_response.strip():
            break

        yield llm_pb2.GenerateReply(message=str(partial_response))

        # 将新生成的文本加入到下次输入中，确保模型使用完整的上下文
        # 将输入转换为新的 input_ids，并更新 attention_mask
        model_inputs = tokenizer(
            text + partial_response,  # 将新生成的部分加入到输入中
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")
        model_inputs.input_ids = model_inputs.input_ids.long()
        attention_mask = model_inputs['attention_mask']


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

    registry_config = RegistryConfig.from_url("zookeeper://mse-791d01018-zk.mse.aliyuncs.com:2181")
    bootstrap = dubbo.Dubbo(registry_config=registry_config)

    service_config = ServiceConfig(service_handler)

    # start the server
    server = bootstrap.create_server(service_config).start()

    input("Press Enter to stop the server...\n")
