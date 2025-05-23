from mistralai import Mistral

API_KEY = "your key"

client = Mistral(api_key=API_KEY)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图片的内容。"},
            {"type": "image_url", "image_url": "https://www.keaitupian.cn/cjpic/frombd/1/253/2537911176/2928325435.jpg"}
        ]
    }
]

response = client.chat.complete(
    model="pixtral-12b-latest",
    messages=messages
)

print(response.choices[0].message.content)
