# Conversational Voice RAG — Persian-first

Independent prototype for **natural spoken conversation**, with RAG as a grounding capability rather than the entire conversational flow.

## What it does

- **Real voice interaction** through microphone and speaker in LiveKit console mode.
- **Interruption / barge-in**: user speech can interrupt ongoing assistant audio; tuned with Silero VAD.
- **Multi-turn memory** in each LiveKit AgentSession, with pronoun/follow-up expansion for search queries.
- **Turn-by-turn retrieval**: every non-social voice turn retrieves from indexed documents (not left to optional LLM tool-calling).
- **Citations**: passages carry source filenames (PDF pages if present), which the assistant can cite verbally.
- **RAG** with offline BM25 retrieval; optional semantic embeddings + reciprocal-rank fusion. Real knowledge is provided by the user.
- **Persian** transcription and spoken responses; models can be changed by configuration.
- **Private-by-default source storage**: index stays local. However, **audio, transcriptions, spoken answers, retrieved excerpts and (when enabled) embedding requests reach the configured OpenAI API**; this is not an offline/private inference stack.

## Requirements

- Python 3.11–3.13, `uv`, working microphone and speaker.
- An **OpenAI API key** for STT, LLM and TTS. An API bill is possible.
- No LiveKit account is required for **console mode** (the initial local interactive experience).
- For browser/mobile/telephone LiveKit rooms, configure LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET and run dev/start. A web frontend is **not included** in this package.

## Run it locally

```bash
uv sync --group dev --extra documents
cp .env.example .env
# Open .env and set OPENAI_API_KEY; don't commit it

# Build the knowledge index from files in ./knowledge
uv run voice-rag-index --source knowledge --output data/knowledge-index.json

# Optional semantic retrieval, sends document chunks to OpenAI for embeddings:
uv run voice-rag-index --source knowledge --output data/knowledge-index.json --semantic

# Download local Silero speech detection model weights:
uv run python src/agent.py download-files

# Talk to it from your terminal microphone / speakers (local, no LiveKit account):
uv run python src/agent.py console

# Inspect retrieval and conversational follow-ups without external API requests:
uv run python -m voice_rag.demo

# Tests (no API calls needed):
uv run pytest -q
```

Tip: use headphones initially to avoid loudspeaker-to-microphone echo causing false interruptions. Console mode also supports text testing.

## Add knowledge

Place UTF-8 `.txt` or `.md`, PDF with extractable text, or `.docx` under `knowledge/`. Re-run `voice-rag-index`. PDFs consisting of scans need an OCR ingestion adapter (not included). Sample knowledge is explicitly fictional, intended only for demonstration.

## How the conversation works

```text
Microphone / LiveKit room
        | streaming audio
Silero VAD -> Persian STT -> user utterance
                                 |
                        conversation state
                                 |
                  context-aware RAG query
                                 |
                  BM25 + optional embeddings
                                 |
                  snippets with source IDs
                                 |
                   conversational LLM
                                 |
               streaming TTS -> speaker
                        ^
           barge-in interrupt loop
```

Design choice: We use **VAD-only endpointing** instead of LiveKit's semantic turn detector, which currently does not list Persian among its supported languages. Fixed 0.8-second endpointing is a starting point, not a validated optimum. Tune using real Persian dialogues and pauses.

## Conversational acceptance scenarios

See [`docs/acceptance.md`](docs/acceptance.md) for hands-on tests focused on naturalness, memory, and interruption handling.

## Known limits (MVP, not production)

- Not end-to-end voice tested in this artifact build (requires credentials, microphone and audio device).
- Per-room conversation memory only; no database-backed long-term memory or user identity/ACL.
- No browser UI or SIP integration bundled yet; LiveKit-compatible deployment extension possible.
- Streaming audio is managed by LiveKit, but STT endpointing and TTS latency vary by provider and network.
- Persian pronunciation, latency, echo, interruptions and noisy environments still require on-device evaluation.
- BM25 is lexical, and optional embeddings improve semantic retrieval. Retrieval confidence cutoffs need evaluation data.
- No streaming audio capture/recording by the app; review data retention/security of chosen external providers separately.

## References

- https://docs.livekit.io/agents/logic/turns/
- https://docs.livekit.io/agents/logic/turns/tuning/
- https://docs.livekit.io/agents/logic/external-data/
- https://docs.livekit.io/agents/models/stt/openai/
- https://docs.livekit.io/agents/models/tts/openai/
- https://docs.livekit.io/agents/server/startup-modes/


### Improve Persian accuracy without sacrificing interim update frequency

For speech in Persian, compare the **finalized** lines (numbered `[1]`, `[2]`, ...)
rather than interim `…` hypotheses. Whisper is a chunked model, not a native
streaming recognizer: intermediate text can be revised. We keep a 300 ms
**partial update request interval**, decode those quick previews with beam=1,
and re-decode the complete utterance with beam=5 (now the default). Final
transcripts may take longer than partials.

A good starting profile on an NVIDIA GPU:

```bash
bash scripts/run-whisper-cuda.sh --model large-v3-turbo --interval-ms 300 --silence-ms 1000 --final-beam-size 5
```

For an alternative accuracy-versus-latency comparison:

```bash
bash scripts/run-whisper-cuda.sh --model medium --interval-ms 300 --silence-ms 1000 --final-beam-size 5
```

Use the **same spoken sample** to compare errors and latency. The best model for
your Persian accent and microphone is empirical, not guaranteed from model size.
If the final text contains incorrect specialized words, add a short vocabulary
hint (do not supply a long instruction):

```bash
bash scripts/run-whisper-cuda.sh --model medium --interval-ms 300 --silence-ms 1000 --initial-prompt "مکالمه، هوش مصنوعی، بازیابی اسناد، مدل زبانی"
```

If quiet words are dropped, try `--vad-mode 1`. If sentences split when you pause,
increase `--silence-ms`; this **adds to finalization delay**, not the interval
between interim update requests. To isolate input quality, test a headset mic,
different input device, and a quiet room. More aggressive decoding cannot fix
a distorted microphone signal.

## Test local Persian real-time speech recognition (no API key)

This is a **separate standalone microphone tool**, not the Voice RAG agent. It uses
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), WebRTC VAD and sounddevice.
It loads a multilingual Whisper model locally, fixes the speech language to Persian
(fa), and displays **revised partial text during speech** plus **final sentences after silence**.

The microphone is sampled at 16 kHz mono PCM; 30 ms frames are analyzed by VAD.
The model receives an updated utterance snapshot **every 300 ms** while you talk;
a dedicated worker runs inference without blocking microphone capture.
Old queued *partial* requests are replaced by the latest one when inference is slow,
but final utterances are never dropped. **300 ms is the requested update interval,
not an end-to-end latency SLA**. Actual text display can be slower on CPU or with
large models. This is pseudo-streaming: partial hypotheses can change until finalized.

### Ubuntu / Linux

```bash
sudo apt update
sudo apt install -y libportaudio2
git pull origin main
uv sync --extra mic
uv run --extra mic voice-rag-whisper --list-devices
uv run --extra mic voice-rag-whisper --model small --device auto --interval-ms 300
```

### GPU: fix missing libcublas.so.12 on Ubuntu

The current CTranslate2 GPU backend requires CUDA 12 cuBLAS and cuDNN 9.
The audio/microphone library `libportaudio2` is unrelated to these GPU dependencies.
Use the self-contained launcher from your repository root:

```bash
git pull origin main
bash scripts/run-whisper-cuda.sh --model small --interval-ms 300
```

The launcher installs the required NVIDIA runtime libraries into the uv virtual
environment and sets `LD_LIBRARY_PATH` **before** launching Python. This requires
a working NVIDIA driver (`nvidia-smi`), Linux x86_64, and enough free disk space
to download the sizeable cuBLAS/cuDNN runtime packages. It does not install
a system-wide CUDA Toolkit. The script checks for missing libraries first.

**Immediate CPU fallback**, with no NVIDIA libraries needed:

```bash
uv run --extra mic voice-rag-whisper --model base --device cpu --compute-type int8 --interval-ms 300
```

Explicit `--device cuda` now fails early with a single actionable error message
if required libraries are unavailable. Unexpected GPU errors during decoding
also stop processing rather than printing the same missing-library error repeatedly.

Source: https://github.com/SYSTRAN/faster-whisper#gpu
If you have multiple microphones, use `--list-devices` then
`--input-device <ID-or-name>`. Save final transcriptions:

```bash
uv run --extra mic voice-rag-whisper --model small --interval-ms 300 --output transcripts.txt
```

Speak Persian naturally, allow a ~750 ms silence between phrases, press **Ctrl+C** to
stop. Use `--silence-ms 1000` if short natural pauses split your sentences.
Use `--vad-mode 1` if quiet speech is ignored (0–3; higher is stricter).
The first run downloads the Whisper model; later runs reuse the local cache.
No OpenAI, OpenRouter or NaraRouter key is required for this standalone test.
Models and microphone samples stay local during inference; initial model download
uses the public Hugging Face hub.
