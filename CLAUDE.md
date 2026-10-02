# CalmGesture — project context (ICT 4442 Deep Learning mini project)

Handoff from a claude.ai session to Claude Code. Read this first.

## What this is
- Course: ICT 4442 (Deep Learning), School of Computer Engineering, MIT Manipal.
- Project: **CalmGesture: A Gesture-to-Sound Sensory Regulation Tool for Autism Support Using Deep Learning.** Webcam hand gestures -> MediaPipe landmarks -> classifier -> predefined calming sounds. Exploratory, non-clinical prototype; never claim diagnosis or therapeutic effect.
- Team (4): Sangini Singh 230911020 (MLP), Tanvi Gupta 230911122 (CNN), Aditya Nigam 230911366 (1D-CNN), Shubh Rastogi 230953258 (GRU). The user is **Aditya Nigam**. (Aditya once wrote "Tanvi Singh"; the approved synopsis says "Tanvi Gupta" — confirm with him.)
- Shared work (all four): dataset prep, MediaPipe processing, common evaluation, custom data recording, final integration.
- Datasets: 20BN-Jester subset (rolling hand, shaking hand, drumming fingers, swiping; needs Qualcomm registration, large) + small custom webcam set (rolling, shaking, waving, finger tapping, open/close). SSBD is related work only.
- Milestones: Interim report (Part B) -> 1D-CNN + GRU trained and 4 models compared by **10 Oct 2026** -> integration, error analysis, Final Report by **31 Oct 2026**. Final report format is IEEE-style (Part C of the template: Times New Roman, 10 pt body, A4, no page numbers, 12+ references, signed contribution page; each member is vivaed individually).

## Repo layout
- `src/preprocess.py` — MediaPipe Hands -> (T, 63) landmarks, wrist-relative + scale-normalised (wrist to middle-finger-MCP distance), resampled to 30 frames, saved as .npy under `data/processed/<class>/`. Input: `data/raw_videos/<class>/*.mp4`.
- `src/utils.py` — loading, `mlp_features` (mean+std over time -> 126-d), stratified split 70/10/20 seed 42, metrics (accuracy, macro-F1, confusion matrix).
- `src/train_mlp.py` — sklearn MLPClassifier (64, 32), standardised features. (Sangini's model)
- `src/cnn_numpy.py`, `src/train_cnn.py` — from-scratch NumPy 2D CNN on a 2x21x3 (mean, std) pose image. (Tanvi's model)
- `src/synth_smoketest_data.py` — procedural synthetic landmark clips. **Pipeline check only.**
- `results/*.json` — metrics from the synthetic run.
- `build_report.js` — generates `ICT4442_Interim_Report_CalmGesture.docx` (`npm install docx`; Times New Roman, black and white).

## Honest status (do not blur this)
- Done: pipeline code, MLP, CNN, interim report draft, literature table (10 papers, details taken from abstracts).
- **All reported metrics are from synthetic data** (200 clips, 5 classes x 40): MLP 95.1% / CNN 100% test accuracy on 41 clips. They only prove the code runs. Never present them as project results.
- `preprocess.py` has **never been run on real video**. It uses the legacy `mp.solutions.hands` API; on newer MediaPipe versions this may be missing, so be ready to switch to the Tasks API (HandLandmarker).
- Real Jester subset and custom recordings: not yet obtained/processed.
- No PyTorch/TensorFlow was available in the sandbox where this was written, so the CNN is NumPy-only. A PyTorch port (ideally adapted from Gesture-Symphony) is still to do.
- Observation worth keeping: a CNN on the mean pose alone got only 41.5% on the synthetic split; the classes differ in motion, not static pose. Adding a temporal-std channel fixed it. This motivates the temporal models.

## Next steps (priority order)
1. Record custom clips and obtain the Jester subset; run `preprocess.py` on a few clips first, fix any MediaPipe API issue, then process everything. Log per-clip hand-detection rate.
2. Re-run `train_mlp.py` and `train_cnn.py` on real data (same split, same metrics); update the report numbers and the "Data status" paragraph.
3. **Aditya's 1D-CNN:** 1D convolution over the full (30, 63) sequence (time as the axis, 63 channels), same split and metrics, report inference latency. Then Shubh's GRU on the same input. Target 10 Oct.
4. Common comparison table: accuracy, macro-F1, confusion matrix, latency, plus error analysis for at least one model.
5. Real-time app: webcam -> best model -> sound trigger; aim for roughly 25-30 ms inference (Gesture2Music reports 25-30 ms).
6. Set up the GitHub repo with real per-member commits (the interim report requires the link and commit history). Replace the placeholders in the report: repo URL, team number, each member's own "Tasks completed", signatures.

## Ground rules
- Never fabricate results, citations or contributions. Do not write other members' contribution text for them; leave placeholders. Each member will be vivaed on their own part.
- Keep one shared split and metric set across all models.
- Report style: Times New Roman, black and white.
- Literature notes: Zhang et al. 2022 classified **EEG**, not gestures; the Qualcomm Jester page is not a paper, so use Zhang, Liu & Wu 2019 (arXiv:1906.07052) as the Jester benchmark citation. The synopsis (reference list there) predates these corrections.
