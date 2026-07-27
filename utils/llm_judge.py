class LLMJudge:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def judge(self, question, reference, prediction):
        prompt = (
            "Judge whether the prediction correctly answers the question using the reference.\n"
            "Output exactly [[CORRECT]] or [[INCORRECT]].\n\n"
            f"Question: {question}\n"
            f"Reference: {reference}\n"
            f"Prediction: {prediction}"
        )
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        response = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return "[[CORRECT]]" in response.upper()
