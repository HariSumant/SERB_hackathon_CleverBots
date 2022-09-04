#imports
from datasets import load_metric
import re
import pandas as pd

metric = load_metric("rouge") #load rouge metric

df_main = pd.read_csv('val.csv') #reference summaries
df_test = pd.read_csv('output_bart.csv')#predicted summaries
df_test.rename(columns={'0': 'Pred'}, inplace=True)
print(df_test.head())

ref_summaries = list(df_main['RHS'])
ref_summaries = ref_summaries[0:20]
print(ref_summaries)


def calc_rouge_scores(candidates, references): #calculate rouge scores
    result = metric.compute(predictions=candidates, references=references, use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100, 1) for key, value in result.items()}
    return result


for i in range(20):
    candidate_summaries = list(df_test['Pred'].apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)[:i + 1])))
    print(f"First {i + 1} sentences: Scores {calc_rouge_scores(candidate_summaries, ref_summaries)}") #print rouge scores
