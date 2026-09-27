import os

import backoff
import openai

# from config import config

completion_tokens = prompt_tokens = 0

# openai.api_key = config.openai_api_key

api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

def on_giveup(details):
    print(f"Giving up after {details['tries']} tries.")
    return None

@backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=5, on_giveup=on_giveup)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(system_prompt, prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None, json=False) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        if json:
            res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop, response_format={'type': 'json_object'})
        else:
            res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        if res is None:
            return None
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    # if config.verbose:
    #     print(gpt_usage(model))
    return outputs

def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    # elif backend == "gpt-3.5-turbo":
    else:  # assume gpt-3.5
        cost = completion_tokens / 1000 * 0.0015 + prompt_tokens / 1000 * 0.0005
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}