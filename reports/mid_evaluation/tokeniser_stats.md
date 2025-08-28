Starting Quantitative Evaluation: Token Coverage
Analyzing a sample of 500,000 lines from each language file...

Processing 'english'...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500000/500000 [00:52<00:00, 9521.75it/s]

Processing 'hindi'...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500000/500000 [00:58<00:00, 8514.60it/s]

Processing 'sanskrit'...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500000/500000 [01:21<00:00, 6103.87it/s]

```
--- Tokenizer Coverage Report ---
{
    "english": {
        "total_tokens_sampled": 22472682,
        "unknown_tokens_found": 150,
        "unknown_token_rate_%": 0.0007,
        "token_coverage_%": 99.9993
    },
    "hindi": {
        "total_tokens_sampled": 28384828,
        "unknown_tokens_found": 63,
        "unknown_token_rate_%": 0.0002,
        "token_coverage_%": 99.9998
    },
    "sanskrit": {
        "total_tokens_sampled": 50762617,
        "unknown_tokens_found": 18,
        "unknown_token_rate_%": 0.0,
        "token_coverage_%": 100.0
    }
}
```

Full report at `model/tokenizer/coverage_report.json`

--- Starting Qualitative Analysis: Vocabulary Inspection ---

Sample of initial vocabulary (special tokens and common subwords):
  \<pad> <br>
  \<unk> <br>
  \<s> <br>
  \</s> <br>
  ▁t <br>
  ▁a <br>
  in <br>
  he <br>
  re <br>
  on <br>
  ▁क <br>
  ▁the <br>
  er <br>
  ▁s <br>
  ▁w <br>
  at <br>
  ▁o <br>
  nd <br>
  ▁c <br>
  it

Sample of mid-range vocabulary (mix of scripts):
  ▁newest <br>
  ▁profiles <br>
  ?! <br>
  ▁1950 <br>
  छा <br>
  ▁McK <br>
  ▁enemies <br>
  ▁departments <br>
  bes <br>
  ▁ME <br>
  ▁कण <br>
  िकोण <br>
  ▁Tai <br>
  ▁Movie <br>
  ▁सका <br>
  ▁seal <br>
  ▁किस्म <br>
  ▁desert <br>
  ▁keyboard <br>
  ▁medications

Sample of final vocabulary (likely rare subwords):
  🚺 <br>
  🚾 <br>
  🛀 <br>
  🛁 <br>
  🛄 <br>
  🛇 <br>
  🛫 <br>
  🛰 <br>
  🛸 <br>
  🟣 <br>
  🟥 <br>
  🟧 <br>
  🟨 <br>
  🟩 <br>
  🢐 <br>
  🤒 <br>
  🤛 <br>
  🤢 <br>
  🤨 <br>
  🤭

Qualitative check: As we see a mix of English characters and Devanagari script (e.g., 'ार', 'ation') in the samples above, it's a good sign.

Coverage ratio:
1. For english: ~2.2
2. For Hindi: ~1.78
3. For sanskrit: ~1 - 2