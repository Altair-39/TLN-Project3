from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline


def load_pipeline() -> Pipeline:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=True
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.4,
        top_p=0.95,
        do_sample=True
    )
