import openai

openai.api_key = "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"

model = "gpt-3.5-turbo"

temperature = 0.9

message_list = []


system_message = {"role": "system", "content": "You are a summarizer. You are given a text and you must summarize it."}

message_list.append(system_message)

user_message = {"role": "user", "content": ""}