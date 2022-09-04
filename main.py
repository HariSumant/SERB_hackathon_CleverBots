#imports
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
import pandas as pd

#tokenizer and model experiment 1
tokenizer_model_1 = PegasusTokenizer.from_pretrained("google/pegasus-large")
loaded_model_1 = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

#tokenizer and model experiment 2
tokenizer_model_2 = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

#read data
df = pd.read_csv('test.csv')

text = list(df[' "Abstract"'])

r = int(len(text))
print(r)


#generate summary google pegasus
def generate_summary_google(text):
    pred = []
    for i in range(r):
        texts = text[i]
        tokens = tokenizer_model_1(texts, truncation=True, padding="longest", return_tensors="pt")
        summary = loaded_model_1.generate(**tokens)
        prediction = tokenizer_model_1.decode(summary[0])
        pred.append(prediction)

    print(pred)
    out_df = pd.DataFrame(pred)
    out_df.to_csv('output_google_test.csv', index=False)


#generate summary facebook bart
def generate_summary_facebook(text):
    pred = []
    for i in range(r):
        texts = text[i]
        tokens = tokenizer_model_2.batch_encode_plus([texts], return_tensors="pt")
        summary = model.generate(tokens['input_ids'], max_length=150, early_stopping=True)
        prediction = tokenizer_model_2.decode(summary[0], skip_special_tokens=True)
        pred.append(prediction)

    print(pred)
    out_df = pd.DataFrame(pred)
    out_df.to_csv('output_bart_test.csv', index=False)


if __name__ == '__main__':
    #function calling
    generate_summary_google(text)
    generate_summary_facebook(text)
