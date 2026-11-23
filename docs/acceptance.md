# Conversational acceptance tests (manual)

Run with demo FAQ. Test in quiet conditions first, then noisy environments. Record P50/P95 time-to-first-audio, turn errors and STT word error rate before assigning any performance claims.

1. **Greeting and small talk**: say «سلام، خوبی؟»; expect a brief friendly answer, with no random document details.
2. **Grounded fact**: «مدت امانت کتاب چقدره؟»; expect base 14 days and special 21 days, not hallucinated terms.
3. **Follow-up memory**: «برای عضویت ویژه چطور؟»; expect relevant answer using prior question context.
4. **Interrupt the voice**: ask for a lengthy explanation, speak «نه، خلاصه بگو» while it talks; check audio promptly stops and it follows the new instruction.
5. **Natural pauses**: say «می‌خواستم بدونم ... [pause 0.6–1.2s] ... چطوری تمدید کنم؟»; tune VAD endpointing to avoid premature answer.
6. **Out-of-scope**: «آدرس شعبه مشهد کجاست؟»; expect 'not found' rather than inventing an address.
7. **Source evidence**: «این از کدوم سند بود؟»; expect source filename (demo_faq.md), not a fabricated link.
8. **Background noise / TV**: check false interruptions, noise echo and unintended speech capture.
9. **Language**: test colloquial Persian, code-switching, numbers, proper names and dates.
10. **Long session**: 15+ turns; validate state consistency and retrieval relevance.

## Quality targets to measure (not achieved claims)

- Barge-in detection and audio-stop latency, measured end-to-end.
- Time to first audible response after end of utterance (P50/P95).
- Percent of correctly detected conversational turns; false early cuts.
- RAG groundedness and citation accuracy over a labeled Persian question set.
- Context carryover in reference questions and pronouns.
