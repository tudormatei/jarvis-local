# 🚀 Final Benchmark Results

```
================================================================================
FINAL BENCHMARK RESULTS
================================================================================
Rank  Model                TTFT (ms)  Latency (ms)   Tok/s   QA %  Instr %  Struct %
-------------------------------------------------------------------------------------
1     granite4:350m              133            269    26.0    100      33        60
2     qwen3:0.6b                 194           1430     5.6    100     100        60
3     qwen3:1.7b                 203           2447     6.6    100     100        80
4     llama3.2:1b                221            289     8.9    100      50        20
5     llama3.2:3b                244            821    20.9    100      67        60
6     deepseek-r1:1.5b           246            449    17.8     88      33        60
7     gemma3:1b                  415            581    14.4    100      83         0
================================================================================
```

🏆 **Fastest Model:** `granite4:350m` (TTFT: **133 ms**)

---

### 🔎 Notes

- TTFT = Time to First Token (responsiveness)
- Latency = Full completion time
- Tok/s = Tokens per second (generation speed)
- QA / Instr / Struct = Accuracy percentages
