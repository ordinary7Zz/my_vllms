from openai import OpenAI

client = OpenAI(api_key="sk-poe-5KZrVvL1MSRG8T_WLCZvRdkW8zwtJAlaPtXrzEf82eM", base_url="https://api.poe.com/v1")

resp = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "user", "content": "只输出 0 或 1"}
    ],
    max_tokens=16,
    logprobs=True,
    top_logprobs=5,
)

print(resp)
print(resp.choices[0].logprobs)