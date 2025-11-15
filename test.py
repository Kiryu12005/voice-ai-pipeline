from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text)