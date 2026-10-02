const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, HeadingLevel,
} = require("docx");
const fs = require("fs");

const FONT = "Times New Roman";
const W = 9906; // A4 content width in DXA (margins 1000 each side)

const run = (text, o = {}) => new TextRun({ text, font: FONT, size: 21, ...o });

const h1 = (text, o = {}) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  ...o,
  spacing: { before: 200, after: 90 },
  children: [new TextRun({ text, bold: true, size: 24, font: FONT, color: "000000" })],
});

const p = (parts, o = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100, line: 264 },
  ...o,
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)),
});

const bullet = (parts) => new Paragraph({
  numbering: { reference: "bl", level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 50, line: 264 },
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)),
});

const cell = (text, width, o = {}) => new TableCell({
  width: { size: width, type: WidthType.DXA },
  shading: o.header ? { type: ShadingType.CLEAR, fill: "000000" } : undefined,
  margins: { top: 50, bottom: 50, left: 80, right: 80 },
  children: String(text).split("\n").map((line) => new Paragraph({
    spacing: { after: 0 },
    children: [new TextRun({
      text: line, font: FONT, size: o.size || 17, bold: !!o.header || !!o.bold,
      color: o.header ? "FFFFFF" : "000000",
    })],
  })),
});

const table = (widths, header, rows, o = {}) => new Table({
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: widths,
  rows: [
    new TableRow({ tableHeader: true, cantSplit: true, children: header.map((t, i) => cell(t, widths[i], { header: true, size: o.size })) }),
    ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], { size: o.size })) })),
  ],
});

const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const infoLine = (label, value) => new Paragraph({
  spacing: { after: 50 },
  children: [run(label + " ", { bold: true }), run(value)],
});

// ---------------- Literature table ----------------
const lit = [
  ["[1] Rajagopalan, Dhall & Goecke, 2013", "Bag-of-words action recognition baseline", "SSBD: 75 in-the-wild videos; arm flapping, head banging, spinning", "First public dataset of self-stimulatory behaviours; baseline recognition reported (no headline figure in abstract)", "Shows camera-based analysis of repetitive movement is feasible. Related work only; not used for training."],
  ["[2] Mohammadian Rad et al., 2015", "CNN feature learning + SVM on accelerometer signals; transfer learning", "6 autistic subjects, wrist/torso 3-axis accelerometers, two studies 3 years apart", "Mean F1 (Study 1 / 2): handcrafted 0.54 / 0.40; CNN 0.74 / 0.35; transfer-learning CNN 0.78 / 0.53", "Learned features beat handcrafted ones for repetitive-movement detection. Wearable modality differs from our camera input."],
  ["[3] Mohammadian Rad et al., 2017", "CNN features, LSTM temporal modelling, transfer learning, LSTM ensembles on IMU data", "Simulated data + two real stereotypical-movement datasets", "Feature learning outperformed handcrafted; LSTM ensembles gave more accurate, stable detectors (no figures in abstract)", "Temporal models help on repetitive movement; supports including a GRU in our comparison."],
  ["[4] Zhang et al., 2022", "RNN classifying EEG of children during gesture-interactive-robot music sessions", "10 autistic + 10 healthy children (3-12 y), EEG", "Accuracy 72-94% (mean about 85%); RNN beat CNN (57-61%), LSTM (64-74%), SVM (53-75%)", "Closest precedent linking gesture interaction, music and autistic children. Note: classification was on EEG, not gestures."],
  ["[5] Jeyaraj et al., 2025 (Gesture2Music)", "Causal temporal CNN on body/hand landmarks predicting note-level events", "Custom: 5 volunteers, 21 gesture-note classes, 3,150 clips", "97.9% pitch accuracy; 25-30 ms inference, 60-70 ms loop latency; GRU baseline 94.26%, LSTM 94.7%", "Shows landmark-to-sound is feasible in real time and gives a latency budget; includes a GRU baseline."],
  ["[6] Singh et al., 2024", "MediaPipe 21 landmarks + CNN (3 conv, 3 pool, 2 FC)", "ASL Alphabet: 87,000+ samples, 29 classes", "99.12% accuracy, real time", "Direct precedent for our MediaPipe-landmark + CNN pipeline."],
  ["[7] Gil-Martin et al., 2025", "MediaPipe landmarks + recurrent nets; landmark formats, normalisation and sequence lengths compared", "IPN Hand, subject-wise cross-validation", "Best 84.66 +/- 1.09% (wrist of first frame as reference); 4 landmarks alone still 81.46%", "Informs our normalisation choice; shows cross-user accuracy is far lower than within-user figures."],
  ["[8] Pooja et al., 2025", "GRU-LSTM on hand-gesture sequences", "ASL alphabet (dataset not named in abstract)", "99.39% accuracy; CNN 96.58%, DNN 97.65%, SVM 98.09%, RNN 98.72%", "Supports recurrent models for gesture sequences (our GRU)."],
  ["[9] Zhang, Liu & Wu, 2019", "2D CNN + Temporal Segment Network with Temporal Shift and SE modules on mobile backbones", "Kinetics-400; Jester as case study", "MnasNet+TSM+SE: 93.7% on Jester", "Establishes Jester as a benchmark and shows mobile/real-time feasibility."],
  ["[10] Linardakis et al., 2025", "Survey of 137 papers (2018-2025)", "Multiple benchmark datasets", "Persistent challenges: real-world robustness, occlusion, cross-user generalisation, efficiency", "Frames our four-family comparison; flags cross-user generalisation as a risk."],
];

const refs = [
  "[1] S. S. Rajagopalan, A. Dhall, R. Goecke, \"Self-Stimulatory Behaviours in the Wild for Autism Diagnosis,\" ICCV Workshops, 2013.",
  "[2] N. Mohammadian Rad et al., \"Convolutional Neural Network for Stereotypical Motor Movement Detection in Autism,\" arXiv:1511.01865.",
  "[3] N. Mohammadian Rad et al., \"Deep Learning for Automatic Stereotypical Motor Movement Detection using Wearable Sensors in Autism Spectrum Disorders,\" arXiv:1709.05956, 2017.",
  "[4] Zhang et al., \"The Use of Deep Learning-Based Gesture Interactive Robot in the Treatment of Autistic Children Under Music Perception Education,\" Front. Psychol., 2022, doi:10.3389/fpsyg.2022.762701.",
  "[5] R. Jeyaraj, B. Subramanian, K. Gangadharan, A. Paul, \"Gesture2Music: A Low-Latency Real-Time Framework for Continuous Gesture-Driven Music Generation,\" arXiv:2511.00793.",
  "[6] G. Singh et al., \"Enhancing Sign Language Detection through Mediapipe and Convolutional Neural Networks (CNN),\" arXiv:2406.03729, 2024.",
  "[7] M. Gil-Martin et al., \"Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks,\" SciTePress, 2025, paper 130535.",
  "[8] S. Pooja et al., \"American Sign Language Recognition Using GRU and LSTM,\" SciTePress, 2025, paper 136397.",
  "[9] C.-L. Zhang, X.-X. Liu, J. Wu, \"Towards Real-Time Action Recognition on Mobile Devices Using Deep Models,\" arXiv:1906.07052, 2019.",
  "[10] M. Linardakis, I. Varlamis, G. Th. Papadopoulos, \"Survey on Hand Gesture Recognition from Visual Input,\" arXiv:2501.11992, 2025.",
];

const doc = new Document({
  numbering: { config: [{ reference: "bl", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 240 } } } }] }] },
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { spacing: { before: 200, after: 90 }, outlineLevel: 0 } }],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1000, bottom: 1000, left: 1000, right: 1000 } } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
        children: [new TextRun({ text: "ICT 4442 — Deep Learning Project", bold: true, size: 30, font: FONT })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 6 } },
        children: [new TextRun({ text: "Interim Report (Part B)", bold: true, size: 26, font: FONT })] }),

      infoLine("Team No. and Names:", "CalmGesture (Team No. ____)"),
      infoLine("Members:", "Sangini Singh (230911020), Tanvi Gupta (230911122), Aditya Nigam (230911366), Shubh Rastogi (230953258)"),
      infoLine("Title of the Project:", "CalmGesture: A Gesture-to-Sound Sensory Regulation Tool for Autism Support Using Deep Learning"),
      infoLine("GitHub Repository Link:", "[insert repository URL here — commit history must show each member's contribution]"),

      // 1
      h1("1. Literature Review"),
      p("Ten papers were reviewed across four themes: autism-related repetitive-movement analysis, gesture-to-music interaction, landmark-based gesture classification, and sequence models for gestures. Details below are taken from each paper's abstract."),
      table([1500, 1900, 1750, 2350, 2406], ["Paper (Author, Year)", "Method", "Dataset", "Key Result", "Relevance to Project"], lit, { size: 16 }),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      p("Taken together, the literature supports a MediaPipe-landmark front end feeding lightweight classifiers [6, 7], suggests recurrent or temporal models should help on repetitive motion [3, 5, 8], and warns that cross-user generalisation is the main weakness of small gesture systems [7, 10]. Papers [1]–[4] show that camera- and sensor-based analysis of autism-related movement and music interaction is an active area, but none of them maps recognised hand movements to sound feedback, which is the gap CalmGesture targets."),

      // 2
      h1("2. Dataset Acquisition & Preprocessing"),
      p([run("Datasets. ", { bold: true }), run("(i) A subset of the 20BN-Jester gesture videos (rolling hand, shaking hand, drumming fingers, swiping and similar classes) and (ii) a small custom webcam dataset recorded by the team (rolling, shaking, waving, finger tapping, hand open/close). SSBD is treated as related work only.")]),
      p([run("Pipeline ", { bold: true }), run("(preprocess.py in the repository):")]),
      bullet("Landmark extraction: MediaPipe Hands, one hand, 21 landmarks × (x, y, z) per frame, up to 60 frames per clip. Frames with no detection reuse the last valid landmarks so every clip is a continuous sequence."),
      bullet("Normalisation: landmarks are made wrist-relative and divided by the wrist–middle-finger-base distance, making them invariant to hand position and distance from the camera."),
      bullet("Fixed length: each clip is linearly resampled to 30 frames, giving a (30, 63) array saved as .npy."),
      bullet("Model inputs: the MLP receives the per-landmark mean and standard deviation over time (126 features, standardised). The CNN receives a 2 × 21 × 3 “pose image” (channel 0 = mean pose, channel 1 = temporal standard deviation), standardised per channel. The 1D-CNN and GRU will consume the full (30, 63) sequence."),
      bullet("Split: stratified 70% train / 10% validation / 20% test with a fixed seed (42), shared by all models."),
      p([run("Data status. ", { bold: true }), run("The pipeline code is complete, but the Jester subset and the custom recordings have not yet been processed through it. The results in Section 3 were therefore produced on 200 procedurally generated synthetic landmark sequences (5 classes × 40 clips) in the same (30, 63) format, solely to validate the pipeline end to end. [TEAM: replace with real-data processing details and clip counts once available.]")]),

      // 3
      h1("3. Models Implemented So Far"),
      table([1100, 1500, 2000, 2406, 2900], ["Model", "Owner (Member)", "Status", "Preliminary Metric", "Notes"], [
        ["MLP", "Sangini Singh", "Implemented; run on synthetic pipeline-check data", "Test acc. 95.1%, macro-F1 94.9% (41 test clips)", "scikit-learn MLP, hidden layers (64, 32), ReLU, early stopping; 126-d standardised mean/std features."],
        ["CNN", "Tanvi Gupta", "Implemented; run on synthetic pipeline-check data", "Test acc. 100%, macro-F1 100% (41 test clips); 1.5 s training, 0.06 ms/clip inference", "From-scratch NumPy 2D CNN (2 conv + 2 dense) on the 2×21×3 pose image. PyTorch port and Gesture-Symphony-based architecture still to do."],
        ["1D-CNN", "Aditya Nigam", "Not started", "—", "Planned: 1D convolution over the (30, 63) landmark sequence. Target 10 Oct 2026."],
        ["GRU", "Shubh Rastogi", "Not started", "—", "Planned: GRU over the (30, 63) sequence. Target 10 Oct 2026."],
      ], { size: 17 }),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      p([run("Reading these numbers. ", { bold: true }), run("The synthetic classes are cleanly separable, so near-ceiling scores confirm that the code works but say nothing about accuracy on real gestures; they must not be quoted as project results. One observation is likely to carry over: a first CNN variant that used only the mean pose reached just 41.5% on the same split, because the gesture classes differ in motion rather than static shape. Adding the temporal standard-deviation channel fixed this, which supports testing explicit temporal models (1D-CNN, GRU) on real data. All four models will be re-run on the real Jester subset and custom data under the same split and metrics (accuracy, macro-F1, confusion matrix, inference latency).")]),

      // 4
      h1("4. Individual Contribution Log (to date)", { pageBreakBefore: true }),
      p("Contributions below reflect the work recorded in the project repository to date and the assignments in the approved synopsis. Each member should confirm their entry matches the commit history before signing."),
      table([1900, 1300, 4906, 1800], ["Member Name", "Reg. No.", "Task(s) Completed", "Signature"], [
        ["Sangini Singh", "230911020", "Implemented the MLP baseline (scikit-learn, hidden layers (64, 32), ReLU, early stopping) on 126-d standardised mean/std landmark features; ran training and baseline evaluation (accuracy, macro-F1) on the synthetic pipeline-check data using the shared split. Shared work: dataset preparation, MediaPipe processing pipeline, common evaluation. Remaining: re-run on real data; recording of custom clips.", ""],
        ["Tanvi Gupta", "230911122", "Implemented the CNN as a from-scratch NumPy 2D CNN (2 conv + 2 dense layers) on the 2×21×3 pose image (mean, std, wrist-relative); trained and evaluated it on the synthetic pipeline-check data, including the observation that adding a temporal-std channel was needed to separate motion-defined classes. Shared work: dataset preparation, MediaPipe processing pipeline, common evaluation. Remaining: PyTorch port; re-run on real data; recording of custom clips.", ""],
        ["Aditya Nigam", "230911366", "Shared work to date: dataset preparation and the MediaPipe preprocessing pipeline (wrist-relative, scale-normalised (30, 63) sequences), the common split and metric set used by all models, and drafting of the interim report. 1D-CNN (1D convolution over the (30, 63) sequence, with inference latency) is not yet started; planned for completion by 10 Oct 2026, followed by the cross-model comparison. Remaining: recording of custom clips; final integration.", ""],
        ["Shubh Rastogi", "230953258", "Shared work to date: dataset preparation, MediaPipe processing and common evaluation setup. GRU model over the (30, 63) sequence is not yet started; planned for completion by 10 Oct 2026, followed by temporal integration into the real-time application. Remaining: recording of custom clips; final integration.", ""],
      ], { size: 17 }),

      // 5
      h1("5. Risk / Plan for Remaining Work"),
      p([run("Remaining models: ", { bold: true }), run("1D-CNN (Aditya Nigam) and GRU (Shubh Rastogi), then comparison of all four on real data and integration of the best model into the real-time gesture-to-sound application.")]),
      table([3000, 3700, 3206], ["Risk / Challenge", "Mitigation", "Owner / Target"], [
        ["Real data not yet processed; reported metrics are pipeline checks only", "Run preprocess.py on the Jester subset and custom clips first; re-run MLP and CNN on the real split before any comparison", "All members; first task before 10 Oct"],
        ["Jester is large and needs registration", "Use only 4–5 selected classes and cap clips per class; keep the subset list in the repository", "All members"],
        ["Jester-to-custom domain gap; small custom set; cross-user generalisation [7, 10]", "Subject-wise splits for the custom data; report Jester and custom results separately; light augmentation (noise, time-warp)", "All members"],
        ["MediaPipe misses hands in fast motion or poor light", "Log per-clip detection rate; drop clips below a threshold; record custom data in good light", "All members"],
        ["CNN is currently NumPy-only", "Port to PyTorch in a GPU-enabled environment and adapt the Gesture-Symphony architecture", "Tanvi Gupta; before 10 Oct"],
        ["Real-time latency in the final app", "Measure per-model inference time; target the 25–30 ms range reported in [5]", "Final integration, 31 Oct"],
        ["Non-clinical scope", "State clearly that the system is an exploratory prototype making no diagnostic or therapeutic claims", "All members"],
      ], { size: 17 }),
      new Paragraph({ spacing: { after: 80 }, children: [] }),
      p([run("Timeline (from the synopsis work plan): ", { bold: true }), run("1D-CNN and GRU trained and all four models compared by 10 Oct 2026; best model integrated into the real-time application, error analysis completed and Final Report submitted by 31 Oct 2026.")]),

      h1("References"),
      ...refs.map((r) => new Paragraph({ spacing: { after: 40 }, alignment: AlignmentType.LEFT,
        children: [new TextRun({ text: r, font: FONT, size: 17 })] })),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync("ICT4442_Interim_Report_CalmGesture.docx", b); console.log("ok"); });
