from django.shortcuts import render
from transformers import BartForConditionalGeneration, BartTokenizer,\
     AutoTokenizer, AutoModelForCausalLM
import spacy

def summary_bart_view(request):
    if request.method == 'POST':
        job_description = request.POST.get('job_description')

        # Load BART model and tokenizer
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        # Generate summary using BART model
        inputs = tokenizer([job_description], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(**inputs, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Extract keywords using spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(summary)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ']]

        return render(request, 'chat/summary.html',
                      {'job_description': job_description, 'summary': summary, 'keywords': keywords})

    return render(request, 'chat/summarize_job_description.html')

def generate_resume(request):
    if request.method == 'POST':
        job_description = request.POST.get('job_description')
        keywords = request.POST.getlist('keywords')

        # Concatenate job description and keywords as prompt
        # prompt = (f"Create a real world project with the "
        #           #f"which is related to the given Job Description and keywords : "
        #           f"using the keywords"
        #           #f"{job_description} | "
        #           f"{', '.join(keywords)}\nstart_response: ")
        prompt = f"Generate work experience based on the following job description {job_description} {keywords} \nstart_response: "


        # Load GPT-J model and tokenizer
        model_name = "google/gemma-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Generate output
        outputs = model.generate(input_ids, max_length=1500, num_return_sequences=1, early_stopping=True)

        # Decode generated work experience
        work_experience = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Process the response and return
        return render(request, 'chat/resume.html', {'work_experience': work_experience})
    else:
        return render(request, 'chat/summary.html')
