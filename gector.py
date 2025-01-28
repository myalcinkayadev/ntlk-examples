from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def correct_grammar(text, max_length=128, num_beams=5, use_sampling=False, temperature=0.9):
    input_text = f"Fix grammatical errors in this text: {text}"
    inputs = tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        do_sample=use_sampling,
        temperature=temperature if use_sampling else None,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def batch_correct(texts):
    input_texts = [f"Fix grammatical errors in this text: {t}" for t in texts]
    inputs = tokenizer(
        input_texts,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    outputs = model.generate(**inputs)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def batch_correct_paragraph(paragraph):
    sentences = paragraph.split(". ")
    corrected = batch_correct(sentences)
    return " ".join(corrected)

# She doesn’t like apples.
print(correct_grammar("She don't likes apples.", use_sampling=False))

# Despite the weather, we decided to go out and eat dinner.
print(correct_grammar("Despite of the whether, we decided to going out and ate dinner.", use_sampling=False))

# The list of items, including all the reports and the data, is missing from the server.
print(correct_grammar("The list of items, including all the reports and the data, are missing from the server.", use_sampling=False))

# When Sarah gave her mother the gift, her mother smiled brightly.
print(correct_grammar("When Sarah gave her mother the gift, she smiled brightly.", use_sampling=False))

# If he had arrived earlier, he would have caught the train.
print(correct_grammar("If he would have arrived earlier, he would had caught the train.", use_sampling=False))

# The team is working on the project. They are trying to finish it by Friday.
# But there are many challenges. Each member has different opinions.
paragraph = "The team is working on the project. Their trying to finish it by Friday. But there’s many challenges. Each member have different opinions."
print(batch_correct_paragraph(paragraph))
