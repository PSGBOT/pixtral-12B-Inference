from mistralai import Mistral
import os

API_KEY = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=API_KEY)

response = client.chat.complete(
    model="pixtral-12b-latest",
    messages=[
        {"role": "user", "content": "请用中文简单介绍一下牛顿第一定律。"}
    ]
)

print(response.choices[0].message.content)
