# One-minute demo script

## Before class

1. Run `start_demo.command` and confirm that all four chapters open.
2. Leave the browser and terminal open. The replay works offline.
3. Use **Full screen**; increase browser zoom if the projector is hard to read.
4. Do not rerun training during the presentation.

## 0–15 seconds: Start small

> We have 737 sports articles, but we supply only two correct answers for each sport. There are ten labeled training articles and 360 other training articles whose answers are hidden. Another 367 articles are reserved for testing.

Show the ten known examples. Explain that an answer is simply a sport, such as football or tennis.

## 15–35 seconds: Watch learning

Open chapter 2. Press **Play** or advance two or three rounds.

> Each computer reader sees a different group of words. They help one another by suggesting possible answers. Because guesses can be wrong, the method keeps several answer sheets and learns how much influence each one should have.

The colored tiles show real guesses. The answer-sheet table shows actual alternatives and learned coefficient shares. If nothing changes in one round, say that the guesses are settling; do not claim every round improves.

## 35–50 seconds: Try an article

Open chapter 3. Select **Find a changed guess** if useful. This chooses a test example where the prediction changed, without requiring that it became correct.

> This article was kept out of training. These are the words available to each reader. This was the starting prediction, and this is the prediction after learning. Let's reveal the correct answer.

Press **Reveal the real answer**. Describe the actual outcome. A remaining error is a valid observation.

## 50–70 seconds: Compare results

Open chapter 4.

> These bars show our measured ranking scores across the declared runs. A good ranking puts the requested sport near the top. This score is different from the percentage of articles classified correctly. We keep the paper's published results in a separate comparison.

Use the actual displayed results. The two-layer method is not guaranteed to beat every alternative in every run.

## Useful Q&A answers

- **Did you use the hidden answers to train?** No. Learners receive the ten supplied answers and word features; the other answers are used only for evaluation and reveal buttons.
- **Is this a real implementation?** Yes. It constructs input-output kernels from alternative labelings and optimizes flat/grouped kernel weights. The demo replays saved predictions from that code.
- **Are these the authors' exact results?** No. This is an independent reproduction with declared random seeds and choices for settings that the paper does not fully report.
- **Why no live training?** Saved rounds make the class demo fast and reliable. The training command is included and can be run separately.
- **Do the two readers see different articles?** No. They see different word features from the same articles.
- **What does influence mean?** It is a candidate's share of learned kernel coefficient weight. It is not the probability that the candidate is correct.

## If something goes wrong

- Browser closed: reopen `start_demo.command`, or visit `http://127.0.0.1:8765` if the terminal is still running.
- Address occupied: stop the other demo with Ctrl+C, or use `python run_demo.py --port 8766`.
- Article headlines unavailable: the predictions and word clues still work. Run `python download_data.py` while online before class to restore the source text.
- No internet in class: the demo and all predictions work locally after setup.
