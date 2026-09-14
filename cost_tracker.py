EMBEDDING_PRICE_PER_TOKEN_USD = 0.02 / 1_000_000
USD_TO_INR = 83.0  # approximate, update if you want precision


class CostTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []

    def record(self, response, call_type):
        usage = response.usage
        if call_type == "embedding":
            cost_inr = usage.prompt_tokens * EMBEDDING_PRICE_PER_TOKEN_USD * USD_TO_INR
            completion_tokens = 0
        else:
            cost_inr = getattr(usage, "cost", 0.0)
            completion_tokens = usage.completion_tokens

        self.calls.append({
            "call_type": call_type,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_inr": cost_inr,
        })

    def totals(self):
        return {
            "num_calls": len(self.calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "cost_inr": sum(c["cost_inr"] for c in self.calls),
        }