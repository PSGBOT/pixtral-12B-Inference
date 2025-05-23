from mistralai import Mistral

API_KEY = "your key"

client = Mistral(api_key=API_KEY)

response = client.chat.complete(
    model="pixtral-12b-latest",
    messages=[
        {"role": "user", "content": "请用中文简单介绍一下牛顿第一定律。"}
    ]
)

print(response.choices[0].message.content)
