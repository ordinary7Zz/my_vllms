gemini 3.5 flash 输出：
```json
RAW RESPONSE OBJECT (first call):
{
  "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOnZlcnRleF9haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOlpxZ2Zhcno4RDZuRnNid1AzdFdTaVE4",
  "created_at": 1780459622.0,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "metadata": {},
  "model": "gemini-3.5-flash",
  "object": "response",
  "output": [
    {
      "id": "Zqgfarz8D6nFsbwP3tWSiQ8",
      "content": [
        {
          "annotations": [],
          "text": "",
          "type": "output_text",
          "logprobs": []
        }
      ],
      "role": "assistant",
      "status": "incomplete",
      "type": "message"
    }
  ],
  "parallel_tool_calls": false,
  "temperature": 0.0,
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "background": false,
  "completed_at": null,
  "conversation": null,
  "max_output_tokens": null,
  "max_tool_calls": null,
  "previous_response_id": null,
  "prompt": null,
  "prompt_cache_key": null,
  "prompt_cache_retention": null,
  "reasoning": null,
  "safety_identifier": null,
  "service_tier": "default",
  "status": "incomplete",
  "text": {
    "format": {
      "type": "text"
    },
    "verbosity": null
  },
  "top_logprobs": 0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 1152,
    "input_tokens_details": {
      "cached_tokens": 0,
      "text_tokens": 72
    },
    "output_tokens": 124,
    "output_tokens_details": {
      "reasoning_tokens": 119,
      "text_tokens": 5
    },
    "total_tokens": 1276
  },
  "user": null,
  "provider_specific_fields": {
    "traffic_type": "ON_DEMAND"
  },
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "store": true
}
```

gemini 3.1 pro 输出：
```json
RAW RESPONSE OBJECT (first call):
{
  "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOnZlcnRleF9haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOllwNGZhcV9oQjV5S2xPb1B3NDJNNkFJ",
  "created_at": 1780457058.0,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "metadata": {},
  "model": "gemini-3.1-pro-preview",
  "object": "response",
  "output": [
    {
      "id": "Yp4faq_hB5yKlOoPw42M6AI",
      "content": [
        {
          "annotations": [],
          "text": "```json\n{\"prediction\": 1, \"confidence\": 0.85}\n```",
          "type": "output_text",
          "logprobs": []
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "parallel_tool_calls": false,
  "temperature": 0.0,
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "background": false,
  "completed_at": 1780457058.0,
  "conversation": null,
  "max_output_tokens": null,
  "max_tool_calls": null,
  "previous_response_id": null,
  "prompt": null,
  "prompt_cache_key": null,
  "prompt_cache_retention": null,
  "reasoning": null,
  "safety_identifier": null,
  "service_tier": "default",
  "status": "completed",
  "text": {
    "format": {
      "type": "text"
    },
    "verbosity": null
  },
  "top_logprobs": 0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 1152,
    "input_tokens_details": {
      "cached_tokens": 0,
      "text_tokens": 72
    },
    "output_tokens": 20,
    "output_tokens_details": {
      "reasoning_tokens": 0,
      "text_tokens": 20
    },
    "total_tokens": 1172
  },
  "user": null,
  "provider_specific_fields": {
    "traffic_type": "ON_DEMAND"
  },
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "store": true
}
```

gpt5.5 输出:
```json
RAW RESPONSE OBJECT (first call):
{
  "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpOb25lO3Jlc3BvbnNlX2lkOnJlc3BfMGYzMjRkMWEzZmQxODAwMjAxNmExZjljNzIxMzEwODE5Yzk3ZDRiOWI0OWE3NDNjZjk=",
  "created_at": 1780456562.0,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "metadata": {},
  "model": "gpt-5.5-2026-04-23",
  "object": "response",
  "output": [
    {
      "id": "msg_0f324d1a3fd18002016a1f9c72dd54819cafa396b19f460fd1",
      "content": [
        {
          "annotations": [],
          "text": "{\"prediction\":1,\"confidence\":0.72}",
          "type": "output_text",
          "logprobs": []
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message",
      "phase": "final_answer"
    }
  ],
  "parallel_tool_calls": true,
  "temperature": 1.0,
  "tool_choice": "auto",
  "tools": [],
  "top_p": 0.98,
  "background": false,
  "completed_at": 1780456563.0,
  "conversation": null,
  "max_output_tokens": 128,
  "max_tool_calls": null,
  "previous_response_id": null,
  "prompt": null,
  "prompt_cache_key": null,
  "prompt_cache_retention": "24h",
  "reasoning": {
    "effort": "none",
    "generate_summary": null,
    "summary": null,
    "context": "current_turn"
  },
  "safety_identifier": null,
  "service_tier": "default",
  "status": "completed",
  "text": {
    "format": {
      "type": "text"
    },
    "verbosity": "medium"
  },
  "top_logprobs": 0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 894,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 15,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 909
  },
  "user": null,
  "store": false,
  "billing": {
    "payer": "developer"
  },
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

