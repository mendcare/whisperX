import os, json, gc, whisperx, torch

# ---------------------  USER SETTINGS  ---------------------
device        = "cuda"
audio_file    = "sample1.mp4"
batch_size    = 16                 # ↓ GPU memory → lower this
compute_type  = "float16"          # "int8" if you’re tight on VRAM
huggingface_token = "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"   # for speaker-diarisation model
swedish_language_code     = "sv" # Language code
# -----------------------------------------------------------

# Helpers – figure out the output folder & stub filenames
base_dir   = os.path.dirname(os.path.abspath(audio_file))
stem       = os.path.splitext(os.path.basename(audio_file))[0]
json_path  = os.path.join(base_dir, f"{stem}_whisperx.json")
txt_path   = os.path.join(base_dir, f"{stem}_transcript.txt")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(torch.backends.cudnn.allow_tf32, "E de do addehhh")

# 1) Transcribe
model   = whisperx.load_model("large-v3", device, compute_type=compute_type, language=swedish_language_code)
audio   = whisperx.load_audio(audio_file)
result  = model.transcribe(audio, batch_size=batch_size)
print("Before alignment:", result["segments"])

# (optional) free up GPU
gc.collect(); del model

# 2) Alignment
model_a, meta = whisperx.load_align_model(language_code=result["language"],
                                          device=device)
result = whisperx.align(result["segments"], model_a, meta, audio, device,
                        return_char_alignments=False)
print("After alignment :", result["segments"])
gc.collect(); del model_a

# 3) Diarisation
diarise = whisperx.diarize.DiarizationPipeline(use_auth_token=huggingface_token,
                                               device=device)
diar_segments = diarise(audio)
result = whisperx.assign_word_speakers(diar_segments, result)

print("Diarisation     :", diar_segments)
print("Final segments  :", result["segments"])

# ---------------- SAVE OUTPUTS ----------------
# a) Full JSON result (start/end, words, speakers…)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# b) Human-readable transcript in chronological order
with open(txt_path, "w", encoding="utf-8") as f:
    for seg in result["segments"]:
        spk  = seg.get("speaker", "SPEAKER_??")
        text = seg["text"].strip()
        f.write(f"{spk} - {text}\n")

print(f"\nSaved JSON   ➜  {json_path}")
print(f"Saved text   ➜  {txt_path}")