# Demonstration Video Script (3–5 minutes)

**Requirement (spec 9f):** a screen recording showing the trained model being loaded and generating text from several prompts. At least one prompt must differ from the training examples, and the actual program must be running.

## Setup before recording

- Open a terminal in the repository folder and make it full-screen, with a large font (about 16–18 pt).
- Activate the environment that has `torch` installed. Locally that is `.venv\Scripts\activate`.
- Run each command once beforehand, so the first load is not slow on camera.
- Keep [README.md](../README.md) open in a second window to show briefly.
- Every model answer listed below was produced by the actual model with `--temperature 0`, so it repeats exactly. Sampled answers (temperature 0.8) will differ from run to run.

---

## 0:00–0:30 · Introduction (talking, show the repo)

> "This is our Mini GPT: an 11.5-million-parameter decoder-only Transformer that we built from scratch in PyTorch. We pretrained it on a synthetic corpus about the residential waste guide of a fictional town, Tidewell, and then fine-tuned it to answer residents' questions. Both trained models are in the `models` folder."

On screen, show:
- the `models/` folder
- `models/model_info.json` (configuration and checksums)

## 0:30–1:30 · Pretrained model: text continuation

```bash
python -m src.generate --ckpt models/pretrained.pt --temperature 0 --max-new-tokens 50 --prompt "Glass bottles and jars go in the"
```

> "The loading line shows the checkpoint, its 11.5 million parameters, 6 layers and the 2,048-token vocabulary. The pretrained model completes a familiar statement from its corpus correctly: blue cart, empty, rinse, lid back on. Then it keeps writing the next document, because it was only trained to predict the next token."

```bash
python -m src.generate --ckpt models/pretrained.pt --temperature 0 --max-new-tokens 60 --prompt "Dear resident, thank you for your question about cooking grease."
python -m src.generate --ckpt models/pretrained.pt --temperature 0.8 --seed 1 --max-new-tokens 60 --prompt "Dear resident, thank you for your question about cooking grease."
```

> "This prompt starts a letter, a framing the model has not seen in training. With greedy decoding it writes correct domain facts: green cart, never down the drain, large amounts to a drop-off depot. With sampling at temperature 0.8, the wording varies and errors creep in. Here it starts with *'Ex machines go to a textile drop-off bin'* before recovering. Higher temperature means more variety, but less reliability."

## 1:30–3:30 · Fine-tuned model: question answering (interactive)

```bash
python -m src.generate --ckpt models/finetuned.pt --qa --temperature 0
```

Type these questions one at a time and comment on each:

| # | Question | What it shows | Expected answer |
|---|---|---|---|
| 1 | `Where do pizza boxes go?` | a basic item question | green organics cart, scrape off food |
| 2 | `What do I do with old paint?` | a hazardous item | hazardous waste drop-off, original container |
| 3 | `When should I put my carts out for collection?` | a topic fact | evening before or early that morning |
| 4 | `Where do almond milk cartons go?` | **an item never seen in training** (held out) | blue cart, rinse, cap on: generalizes through "milk cartons" |
| 5 | `How do I dispose of an old yoga mat?` | **another held-out item** | garbage: correct cart via "mat/plastic", though the prep tip is borrowed from glass |
| 6 | `Can you recommend a good pizza place?` | an off-topic question | polite decline |
| 7 | `We just finished a party and have lots of empty beer cans. Where do they go?` | **an honest failure**: a long, chatty question | an answer about *lancets*. Explain that short, direct questions work best, which matches the report's limitations |

Suggested commentary:
- After 4: "Almond milk cartons never appeared in any training data. The model generalizes through the familiar words 'milk cartons'."
- After 7: "Long, chatty questions confuse this small model, so it latches onto the wrong item. This matches what we found in our evaluation."

Type `quit` to exit.

## 3:30–4:15 · Before vs. after fine-tuning (same question, both models)

```bash
python -m src.generate --ckpt models/pretrained.pt --temperature 0 --max-new-tokens 60 --prompt "Q: Can I put my old electric kettle in the garbage?
A:"
python -m src.generate --ckpt models/finetuned.pt --qa --temperature 0 --prompt "Can I put my old electric kettle in the garbage?"
```

> "On our 654-question evaluation set, fine-tuning raised accuracy from 76% to 82%. The clearest gains were yes/no questions, answering about the item that was actually asked, and declining off-topic questions."

Electric kettles are a held-out item. The expected results, taken from the evaluation run:
- **Pretrained:** *"Yes! … televisions and monitors go to an electronics drop-off."* The yes is wrong and it talks about a different item.
- **Fine-tuned:** *"No, sorry. They count as electronic waste."*

If the pretrained output differs on camera, say so; the evaluation numbers come from the saved evaluation run.

## 4:15–4:45 · Wrap-up

> "To summarize: the pretrained model learned the domain's facts and writing styles. Fine-tuning taught it how to answer: stay on the question, get yes/no right, stop and decline off-topic questions. It did not add new knowledge about unseen items. The code, data, both checkpoints and the full results are in the repository."
