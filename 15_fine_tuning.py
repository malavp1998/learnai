"""
STEP 15: FINE-TUNING — actually changing a model's WEIGHTS with your own
data, instead of changing what you SEND it (prompting) or what you FETCH
for it (RAG, files 06-11). The one technique in this whole project that
doesn't run by calling an API in a loop — it trains.

WHY THIS FILE EXISTS, AND WHY IT LOOKS DIFFERENT FROM 01-14:
Every other file in this project runs top to bottom on your laptop in a
few seconds, because they all just SEND requests to an already-trained
model (Groq's API). Fine-tuning trains — it needs a GPU, real memory,
and minutes-to-hours, not seconds. This laptop is a 2018 Intel MacBook
Pro with no GPU: technically possible to fine-tune on CPU, but slow
enough (hours for a toy example) that it isn't a good way to learn the
concept. So this file is written to be copy-pasted into a FREE Google
Colab GPU notebook (colab.research.google.com -> Runtime -> Change
runtime type -> T4 GPU, free tier) — the code is 100% real and correct,
not pseudocode, you just run it there instead of `python 15_fine_tuning.py`
locally. Cells are marked "COLAB CELL N" — paste each one into its own
Colab cell, in order.

THE WALL THAT MAKES FINE-TUNING NECESSARY (why RAG isn't always enough):
Files 06-11 spent a lot of effort making RAG retrieve the RIGHT text and
paste it into a prompt. That fixes "the model doesn't know your data" —
but it does NOT fix:
  - The model doesn't know your data's STYLE/FORMAT/TONE (e.g. always
    respond in a specific JSON schema, or in your company's voice).
  - The model doesn't know your data's REASONING PATTERN (e.g. how a
    senior engineer at your company reviews code, not just facts to cite).
  - Every single request pays the cost of resending a big context block
    of retrieved chunks — fine-tuning bakes the knowledge into the
    WEIGHTS, so a fine-tuned model can behave correctly on a SHORT
    prompt, no retrieval step needed at inference time.
Fine-tuning trades "flexible, cites sources, easy to update" (RAG) for
"bakes behavior in permanently, no retrieval step, but stale the moment
your data changes and requires retraining."

THE THREE WAYS TO CHANGE MODEL BEHAVIOR, RANKED BY EFFORT (memorize this
ordering — it's the most common fine-tuning interview question):
  1. PROMPTING       — cheapest, fastest, most flexible. Try this first,
                        always. (Files 01-05 of this project.)
  2. RAG             — when the model needs FACTS it wasn't trained on,
                        and those facts change over time. (Files 06-11.)
  3. FINE-TUNING     — when the model needs a NEW SKILL/FORMAT/STYLE that
                        no amount of prompting reliably produces, and
                        that skill is stable (won't change every week).
Fine-tuning is the LAST resort, not the first idea — it's the most
expensive to build, hardest to update, and easiest to get wrong (see
"catastrophic forgetting" and "overfitting" below). Interviewers ask
this specifically to see if you reach for fine-tuning by default (bad
sign) or only when prompting+RAG genuinely can't do the job (good sign).

FULL FINE-TUNING vs LoRA (the mechanism this file actually uses):
  FULL fine-tuning: update EVERY weight in the model. For a 7-billion
  parameter model, that's 7 billion numbers to store gradients+optimizer
  state for — needs 80GB+ of GPU memory, completely out of reach for a
  free Colab GPU (which gives you ~15GB).

  LoRA (Low-Rank Adaptation): freeze the ENTIRE original model, and
  instead train a tiny pair of new, small matrices that get ADDED to a
  few of the original layers. The original model is untouched; LoRA
  learns a small "correction" on top of it. This is why LoRA can train
  on a free Colab GPU in minutes what full fine-tuning couldn't do
  without a multi-GPU server: you're training maybe 0.1-1% of the
  original parameter count.

QLoRA (what this file actually runs): LoRA, PLUS the frozen base model
is loaded in 4-bit quantization (compressed weights) instead of full
16/32-bit precision — cuts memory further, which is what makes even a
7B-parameter model fit on a free T4 GPU's ~15GB at all.
"""

# ============================================================
# COLAB CELL 1 — install dependencies (Colab-only; these are NOT in this
# repo's requirements.txt because they need a GPU to be useful at all)
# ============================================================
# !pip install -q transformers peft bitsandbytes accelerate trl datasets

# ============================================================
# COLAB CELL 2 — build the training dataset
# ============================================================
# WHAT A FINE-TUNING DATASET ACTUALLY IS: a list of (input, desired
# output) EXAMPLES — you're showing the model "when you see something
# shaped like THIS, respond shaped like THAT," many times over, and
# training nudges the weights toward producing that pattern more often.
#
# This example teaches a small model a STYLE/FORMAT skill prompting
# alone struggles to enforce consistently: always answer HR policy
# questions in a strict "Answer: ... | Policy section: ..." format,
# using this project's own docs/company_leave_policy.txt as the source
# domain — the same document 06_rag_basics.py retrieves from, so you
# can directly compare the RAG answer vs. the fine-tuned answer to the
# SAME question later in this file.

TRAINING_EXAMPLES = [
    {
        "instruction": "How many days of paid leave do full-time employees get?",
        "response": "Answer: 18 days of paid leave per year. | Policy section: Annual Leave",
    },
    {
        "instruction": "How do I request leave?",
        "response": "Answer: Submit a leave request at least 3 days in advance through the HR portal. | Policy section: Leave Request Process",
    },
    {
        "instruction": "What is the maternity leave policy?",
        "response": "Answer: 26 weeks paid maternity leave. | Policy section: Parental Leave",
    },
    {
        "instruction": "What is the paternity leave policy?",
        "response": "Answer: 2 weeks paid paternity leave. | Policy section: Parental Leave",
    },
    {
        "instruction": "Can unused leave be carried over to next year?",
        "response": "Answer: Up to 5 unused days may be carried over to the following year. | Policy section: Leave Carryover",
    },
    # A REAL fine-tuning run needs HUNDREDS to THOUSANDS of examples like
    # this, not 5 — this is a toy set so the notebook trains in minutes
    # on a free GPU. With only 5 examples the model will mostly MEMORIZE
    # these exact Q&As rather than learn the general format — a concrete,
    # visible example of the "too little data -> overfitting" problem
    # explained further down this file.
]

# Format each example the way the model expects during training: a single
# text block combining instruction + response with clear delimiters, so
# it learns "text shaped like this -> continue it like that."
def format_example(example: dict) -> str:
    return (
        f"### Question:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    )


# from datasets import Dataset
# formatted = [{"text": format_example(ex)} for ex in TRAINING_EXAMPLES]
# train_dataset = Dataset.from_list(formatted)


# ============================================================
# COLAB CELL 3 — load the base model in 4-bit (QLoRA setup)
# ============================================================
# We fine-tune a SMALL open model here (not Groq's models — Groq only
# serves inference over an API, you cannot fine-tune through it; the
# model whose weights you fine-tune has to be one you can load and
# train directly). TinyLlama-1.1B is small enough to actually finish
# training on a free Colab T4 GPU in a few minutes.

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#
# # 4-bit quantization config — THIS is the "Q" in QLoRA. Loads the base
# # model's weights compressed to 4 bits each instead of the usual 16/32
# # bits, cutting memory ~4-8x. The model still WORKS at 4-bit precision
# # (with a small quality cost) — this is purely a memory optimization to
# # fit training on a free GPU, not part of what "teaches" the model.
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer.pad_token = tokenizer.eos_token
#
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_config,
#     device_map="auto",  # automatically places the model on the GPU Colab gave you
# )


# ============================================================
# COLAB CELL 4 — attach LoRA adapters (the "low-rank" part of LoRA)
# ============================================================
# This is the step that makes LoRA cheap. Instead of unfreezing all
# ~1.1 BILLION of TinyLlama's parameters, we freeze every one of them
# and attach small trainable matrices to a handful of layers
# (target_modules) — ONLY these new small matrices get trained.
#
# r (rank) controls how big those new matrices are — the "how much new
# capacity are we adding" knob. Small r = fewer trainable parameters
# (faster, less memory, more likely to underfit). Large r = more
# trainable parameters (slower, more memory, more likely to overfit on
# a small dataset like ours). r=8 or 16 are common starting points.

# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#
# model = prepare_model_for_kbit_training(model)
#
# lora_config = LoraConfig(
#     r=8,                       # rank — size of the new trainable matrices
#     lora_alpha=16,             # scaling factor for the LoRA update (roughly: how strongly it influences the output)
#     target_modules=["q_proj", "v_proj"],  # WHICH layers get LoRA matrices attached — the attention projection layers, standard choice
#     lora_dropout=0.05,         # regularization — randomly zero out some of the LoRA update during training, reduces overfitting
#     task_type="CAUSAL_LM",
# )
#
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
# # Expect something like: "trainable params: 1.1M || all params: 1.1B ||
# # trainable%: 0.1%" — this line is the entire point of LoRA, made visible:
# # you are training a tenth of a percent of the model's actual weights.


# ============================================================
# COLAB CELL 5 — train
# ============================================================
# This is the ONLY cell that actually changes any numbers. Everything
# before this was setup (loading, formatting, attaching empty LoRA
# matrices); everything after this just USES the result.

# from trl import SFTTrainer
# from transformers import TrainingArguments
#
# training_args = TrainingArguments(
#     output_dir="./tinyllama-leave-policy-lora",
#     num_train_epochs=10,   # how many times the model sees the FULL training set —
#                            # high because our toy dataset is tiny; a real dataset
#                            # of thousands of examples would use far fewer epochs
#     per_device_train_batch_size=2,
#     learning_rate=2e-4,
#     logging_steps=1,
#     save_strategy="no",
# )
#
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     dataset_text_field="text",
#     max_seq_length=256,
#     args=training_args,
# )
#
# trainer.train()
# # Watch the printed "loss" number as this runs — it should generally
# # trend DOWN over steps. Loss = how wrong the model's predictions are
# # on the training examples; a falling loss means it's learning the
# # instruction->response pattern from TRAINING_EXAMPLES above.


# ============================================================
# COLAB CELL 6 — use the fine-tuned model, and compare to the base model
# ============================================================
# The single most important cell to actually LOOK at: run the SAME
# question through the base model (no LoRA) and the fine-tuned model,
# side by side, and see the format difference directly instead of
# taking it on faith.

# def ask(model_to_use, question: str) -> str:
#     prompt = f"### Question:\n{question}\n\n### Response:\n"
#     inputs = tokenizer(prompt, return_tensors="pt").to(model_to_use.device)
#     output = model_to_use.generate(**inputs, max_new_tokens=60, do_sample=False)
#     return tokenizer.decode(output[0], skip_special_tokens=True)
#
# test_question = "How much maternity leave is offered?"
# print("FINE-TUNED MODEL OUTPUT:")
# print(ask(model, test_question))
# # Expect roughly the "Answer: ... | Policy section: ..." format, even
# # though this EXACT question wasn't in TRAINING_EXAMPLES verbatim —
# # that's the difference between memorizing and generalizing the format.
# # With only 5 training examples, don't be surprised if it leans toward
# # memorized phrasing instead — see the overfitting note in Cell 2.


# ============================================================
# WHAT THIS FILE DELIBERATELY LEAVES OUT (read, don't skip):
#
# - EVALUATION: a real fine-tuning project needs a held-out test set
#   (examples the model never trained on) to check it GENERALIZED
#   rather than just memorized the training examples — same principle
#   as 09_retrieval_evaluation.py's golden set, applied to generation.
#
# - CATASTROPHIC FORGETTING: aggressively fine-tuning on a narrow
#   dataset can degrade the model's general abilities on things
#   UNRELATED to your fine-tuning data (it's now biased toward always
#   producing your format/style, even when inappropriate). LoRA's
#   "freeze the original weights" design specifically limits this
#   compared to full fine-tuning, but doesn't eliminate it.
#
# - HYPERPARAMETER TUNING: learning_rate, r, epochs, batch_size all
#   meaningfully affect whether training even converges — real projects
#   run several training runs and compare, not one.
#
# - MERGING LoRA WEIGHTS: for deployment, you can either (a) keep the
#   base model + LoRA adapter as two separate pieces loaded together at
#   inference time (what Cell 6 does), or (b) call
#   `model.merge_and_unload()` to bake the LoRA weights permanently into
#   a new single set of model weights you can deploy standalone. Both
#   are common; (a) is cheaper to store multiple fine-tunes of the same
#   base model, (b) is simpler to deploy/serve.
#
# - RLHF / DPO: this file does SUPERVISED fine-tuning (SFT) — learning
#   from (input, correct output) pairs. RLHF/DPO are a DIFFERENT, more
#   advanced fine-tuning family that instead learns from human
#   PREFERENCES between multiple candidate outputs ("response A is
#   better than response B") — how instruction-following/chat models
#   like the ones this whole project calls via Groq were themselves
#   trained, on top of SFT. Out of scope here, but the natural next
#   question once SFT is understood.
# ============================================================


if __name__ == "__main__":
    print(__doc__)
    print(
        "\nThis file is written for Google Colab's free GPU tier, not for\n"
        "local execution — see the top-of-file docstring for why (no GPU\n"
        "on this machine). Open colab.research.google.com, create a new\n"
        "notebook, set Runtime -> Change runtime type -> T4 GPU, then copy\n"
        "each '# ============ COLAB CELL N ============' block below into\n"
        "its own cell, uncommenting the code, and run them in order."
    )
