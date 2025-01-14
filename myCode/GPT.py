from zhipuai import ZhipuAI
client = ZhipuAI(api_key="5c148c9c641012658e7ada2bc4b02ec8.9IaCmOG6Y2VmjsBN")  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4-plus",  # 请填写您要调用的模型名称
    messages=[
        {"role": "user", "content": "介绍一下功能"},
    ],
)
print(response.choices[0].message)