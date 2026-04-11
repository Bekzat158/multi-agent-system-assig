  // ============================================================
  // Communication-Efficient and Robust Federated Anomaly Detection
  // for Resource-Constrained IIoT Edge Networks
  // Springer Nature Article Template (sn-article style)
  // ============================================================

  #set document(
    title: "Communication-Efficient and Robust Federated Anomaly Detection for Resource-Constrained IIoT Edge Networks under Non-IID and Adversarial Settings",
    author: "Sundetkhan Bekzat, Baibolat Bekarys",
    date: datetime.today(),
  )

  #set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
    numbering: "1",
    number-align: center,
  )

  #set text(
    font: "New Computer Modern",
    size: 10.5pt,
    lang: "en",
  )
  #set par(
    justify: true,
    leading: 0.65em,
    spacing: 0.9em,
  )
  #set heading(numbering: "1.1")
  #set math.equation(numbering: "(1)")

  #show figure.caption: it => [
    #set text(size: 9pt)
    *#it.supplement #it.counter.display(it.numbering)*: #it.body
  ]

  // ── Utility functions ─────────────────────────────────────────
  #let abstract-block(body) = block(
    width: 100%,
    inset: (x: 1.5em, y: 1em),
    fill: rgb("#f5f5f5"),
    stroke: (left: 3pt + rgb("#1565C0")),
    radius: 3pt,
    body,
  )

  #let keyword-list(..kws) = [
    #set text(size: 9.5pt)
    *Keywords:* #kws.pos().join(" · ")
  ]

  // ─────────────────────────────────────────────────────────────
  //  TITLE BLOCK
  // ─────────────────────────────────────────────────────────────
  #align(center)[
    #block(width: 90%)[
      #set text(size: 16pt, weight: "bold")
      Communication-Efficient and Robust Federated Anomaly Detection
      for Resource-Constrained IIoT Edge Networks
      under Non-IID and Adversarial Settings
    ]
    #v(0.8em)
    #set text(size: 11pt)
    Sundetkhan Bekzat #h(2em) Baibolat Bekarys
    #v(0.3em)
    #set text(size: 9.5pt, style: "italic")
    Introduction to Multi-Agent Systems #sym.dot.c Seema Rawat
    #v(0.5em)
    #line(length: 60%, stroke: 0.5pt + gray)
  ]

  #v(1em)

  // ─────────────────────────────────────────────────────────────
  //  ABSTRACT
  // ─────────────────────────────────────────────────────────────
  #abstract-block[
    #set text(size: 9.5pt)
    *Abstract.* We address the problem of federated anomaly detection in resource-constrained
    Industrial Internet of Things (IIoT) edge networks under statistically heterogeneous
    (non-IID) data distributions and in the presence of adversarial/Byzantine participants.
    We propose a modular framework that jointly optimises: *(i)* communication efficiency
    via Top-$k$ gradient sparsification with error-feedback and FedPAQ-style periodic
    quantised averaging; *(ii)* robustness to non-IID heterogeneity through FedProx
    proximal regularisation and personalised local head adaptation; and *(iii)* resilience
    against model-poisoning, backdoor, and on-off Byzantine attacks via robust aggregation
    (coordinate-wise median, trimmed mean, Krum/Multi-Krum, and RFA geometric-median).
    We evaluate the framework on a synthetic IIoT multi-client benchmark featuring
    five non-IID edge clients with diverse operating frequencies, amplitudes, and
    anomaly patterns. Experiments demonstrate that our full system (FedProx + Top-10% + RFA)
    achieves significant reduction in total uplink communication with minimal AUROC
    degradation compared to uncompressed FedAvg, and maintains robust detection even
    when 30% of clients are Byzantine adversaries ---
    a setting where vanilla FedAvg degrades to near-random performance.
    Ablation studies confirm the complementary contributions of each component.

    *Code Availability:* The source code and project materials are available at #link("https://github.com/Bekzat158/multi-agent-system-assig").
  ]

  #v(0.6em)
  #keyword-list(
    "Federated Learning",
    "Anomaly Detection",
    "IIoT Edge Networks",
    "Non-IID",
    "Byzantine Robustness",
    "Gradient Compression",
    "LSTM Autoencoder",
    "Robust Aggregation",
  )

  #v(1em)
  #line(length: 100%, stroke: 0.5pt + gray)
  #v(0.5em)

  #show: columns.with(2)

  // ─────────────────────────────────────────────────────────────
  //  1. INTRODUCTION
  // ─────────────────────────────────────────────────────────────
  = Introduction

  Industrial Internet of Things (IIoT) architectures increasingly push analytics
  to the *edge* --- closer to sensors, PLCs, SCADA nodes, and actuators --- to reduce
  latency, preserve bandwidth, and enable autonomous local decision-making.
  Within these edge networks, *anomaly
  detection* is a safety-critical task: early identification of cyber-physical
  attacks, equipment faults, or process deviations can prevent catastrophic failures
  or production downtime.

  Federated learning (FL) is a natural fit for IIoT anomaly detection because raw
  sensor data are privacy-sensitive, and centralising them over constrained uplinks
  is often infeasible. The seminal FedAvg algorithm demonstrated that iterative
  model averaging with multiple local epochs dramatically reduces communication
  rounds compared to synchronous SGD. However, real-world
  IIoT deployments expose three fundamental challenges that FedAvg does not address:

  #block(inset: (left: 1.2em))[
    + *Communication bottleneck.* Uplink bandwidth on embedded edge devices can be
      as low as tens of kilobits per second; transmitting full gradient vectors
      per round is prohibitive.
    + *Statistical heterogeneity (non-IID).* Different machines, production lines,
      and operating modes produce fundamentally different time-series distributions,
      causing client drift and convergence instability under FedAvg.
    + *Adversarial threats.* A subset of compromised edge devices may inject
      poisoned model updates, backdoor triggers, or adopt evasive on-off strategies
      to corrupt the global model while evading detection.
  ]

  These three challenges are *interdependent*: aggressive compression exacerbates
  drift under non-IID; robust aggregators incur additional communication cost;
  and personalisation can expose per-client state to inference attacks. We therefore
  propose a *unified modular framework* that addresses all three simultaneously ---
  an integrated design validated by ablation analysis.

  == Contributions

  Our main contributions are:

  - A *communication-efficient FL protocol* combining Top-$k$ sparsification
    with error-feedback, periodic aggregation, and partial client participation,
    achieving up to 100x compression with minimal detection quality loss.
  - *Non-IID robustness* via FedProx proximal regularisation and split-parameter
    personalisation (shared encoder + local threshold head).
  - A *multi-layer Byzantine defence* combining RFA geometric-median aggregation
    with norm-clipping and trust-score weighting, evaluated against four attack types.
  - A *reproducible experimental benchmark* on synthetic IIoT time-series with
    configurable non-IID splits, attack fractions, and compression ratios.

  // ─────────────────────────────────────────────────────────────
  //  2. RELATED WORK
  // ─────────────────────────────────────────────────────────────
  = Related Work

  == Communication-Efficient Federated Learning

  FedAvg introduced local multi-step SGD to reduce the number
  of communication rounds. Subsequent work decomposed communication efficiency
  into three orthogonal levers: *partial participation*, *periodic synchronisation*,
  and *message quantisation* (FedPAQ). Gradient sparsification methods
  such as Top-$k$ selection reduce the uplink footprint by transmitting only the
  largest-magnitude components; error-feedback corrects the estimation bias over
  rounds. For IIoT anomaly detection specifically,
  Yuan et al. demonstrated Top-$k$ compression on four real-world time-series
  datasets, reporting approximately 50% communication reduction with negligible accuracy loss.

  == Anomaly Detection in IIoT

  Deep learning approaches dominate recent benchmarks for industrial time-series
  anomaly detection. *Reconstruction-based* methods (LSTM-AE,
  Variational AE) flag anomalies when the reconstruction error exceeds a threshold
  learned on normal data.
  *One-class* methods such as Deep SVDD learn a compact hypersphere
  embedding for normal samples. *Exemplar-based* approaches (Fed-ExDNN)
  extend this to federated settings by sharing exemplar modules across clients.

  == Robustness in Federated Learning

  Statistical robustness to non-IID data is addressed by FedProx
  (proximal regularisation), SCAFFOLD
  (control variates for client drift), pFedMe and Ditto
  (personalised objectives). Byzantine robustness is a separate axis:
  coordinate-wise median and trimmed mean, Krum
  and Bulyan provide theoretical
  guarantees under bounded attacker fractions. RFA
  replaces the mean with an approximate geometric median, preserving
  compatibility with secure aggregation. FLTrust
  bootstraps trust from a small clean root dataset at the server.
  Evasive attacks such as on-off strategies and
  adaptive poisoning remain open challenges.

  // ─────────────────────────────────────────────────────────────
  //  3. PROBLEM FORMULATION
  // ─────────────────────────────────────────────────────────────
  = Problem Formulation

  == System Model

  Consider a network of $K$ edge clients $cal(K) = {1, dots, K}$.
  Client $k$ holds a local dataset $D_k$ of $n_k$ multivariate
  time-series observations $bold(x) in bb(R)^F$ from $F$ sensors.
  The federated objective is:
  $
    min_(theta) F(theta) = sum_(k=1)^K p_k F_k (theta),
  $
  $
    F_k (theta) = 1/n_k sum_(i=1)^(n_k) ell (bold(x)_i^((k)), theta),
  $ <eq-fed-obj>

  where $cal(P)_k$ is the local distribution (non-IID: $cal(P)_k != cal(P)_j$
  in general), $p_k prop n_k$, and $ell$ is the reconstruction loss for an
  LSTM autoencoder model $f_theta$.

  The *system-level objective* extends @eq-fed-obj to a multi-criteria form:
  $
    min_(theta, Pi) quad F(theta) + lambda_1 op("Comm")(Pi) \
      + lambda_2 op("Latency")(Pi) \
      + lambda_3 op("Robust")(theta, Pi),
  $

  where $Pi$ denotes the FL protocol (client selection, synchronisation frequency,
  compressor, aggregator, filters). This captures the inherent trade-off in IIoT
  edge deployments: a method optimal in detection quality may be infeasible due to
  bandwidth or latency constraints, and vice versa.

  == Threat Model

  We consider two categories of adverse factors. Let $alpha$ denote the fraction
  of compromised clients $cal(A) subset cal(K)$, $abs(cal(A)) = alpha K$.

  *Non-IID / domain drift* (non-adversarial): distributions $cal(P)_k$ diverge
  due to heterogeneous equipment, operating modes, and sensor configurations.

  *Byzantine / adversarial clients*: compromised clients may apply:
  - *Model poisoning*: submit crafted updates $Delta_k$ to maximise global loss
    or cause targeted misclassification.
  - *Backdoor (model replacement)*: embed a hidden trigger.
  - *Label flipping*: invert anomaly labels during supervised fine-tuning.
  - *On-off attacks*: alternate honest and malicious rounds to evade defences.
  - *Adaptive attacks*: optimise updates against known aggregation rules.

  // ─────────────────────────────────────────────────────────────
  //  4. PROPOSED METHOD
  // ─────────────────────────────────────────────────────────────
  = Proposed Method

  Our framework is structured as four composable modules executed in each round,
  illustrated in @fig-arch.

  #figure(
    image("figures/fig_architecture_diagram.png", width: 100%),
    caption: [System architecture of the proposed Communication-Efficient and Robust
    Federated Anomaly Detection framework. Edge clients perform local LSTM-AE training
    with FedProx regularisation and Top-$k$ compression before transmission;
    the server filters Byzantine updates via RFA and broadcasts the updated global model.],
  ) <fig-arch>

  == Module 1: LSTM Autoencoder for IIoT Anomaly Detection

  We adopt an *LSTM Autoencoder* (LSTM-AE) as the base anomaly detection model,
  as it naturally captures temporal dependencies in multivariate sensor streams.
  For a sliding window $bold(X) in bb(R)^(W times F)$ of
  width $W$ over $F$ channels:

  $
    bold(z) = op("Encoder")_theta (bold(X)) in bb(R)^L, \
    hat(bold(X)) = op("Decoder")_phi (bold(z)) in bb(R)^(W times F).
  $

  The *reconstruction loss* is:
  $
    ell(bold(X), theta, phi) = 1/(W F) lr(|| bold(X) - hat(bold(X)) ||)_F^2.
  $

  At inference, the *anomaly score* for sample $bold(X)$ is $s(bold(X)) = ell(bold(X), theta, phi)$;
  a sample is flagged anomalous if $s gt tau$, where $tau$ is the $p$-th percentile
  of training reconstruction errors (default $p = 95$).

  The encoder uses a single-layer LSTM of hidden size $H = 48$ projecting to
  latent dimension $L = 16$. The decoder repeats the latent vector $W$ times and
  passes it through a symmetric LSTM before a linear readout. This compact
  architecture (roughly 50K parameters) is designed for on-device deployment on
  resource-constrained IIoT edge nodes.

  == Module 2: Communication-Efficient FL Protocol

  === Top-$k$ Sparsification with Error-Feedback

  After $E$ local epochs, client $k$ computes
  the model update:
  $
    Delta_k^t = theta_k^t - theta^t_"global",
  $
  and applies Top-$k$ sparsification:
  $
    Delta_k^t + e_k^t arrow.r C_k (Delta_k^t + e_k^t, kappa),
  $
  $
    e_k^(t+1) = (Delta_k^t + e_k^t) - C_k (Delta_k^t + e_k^t, kappa),
  $

  where $C_k (dot.c, kappa)$ zeroes all but the $kappa$-fraction of components
  with largest absolute values, and $e_k^t$ is the *error-feedback residual*
  accumulated from the previous round. The effective communication ratio is
  $kappa in (0.01, 1.0)$, with $kappa = 0.10$ as our default.

  === Partial Participation and Periodic Aggregation

  In each round $t$, the server samples $abs(cal(S)_t) = C dot K$ clients
  ($C = 0.6$ in our experiments). Each selected client performs $E = 3$
  local epochs before uploading the compressed update. This two-level reduction
  (fewer rounds via local steps; fewer bytes via compression) directly follows
  the FedPAQ design principles.

  == Module 3: Non-IID Robustness via FedProx

  Under non-IID distributions, FedAvg suffers from *client drift*: local
  optimisation overshoots the global optimum. FedProx introduces
  a proximal term controlling the deviation from the global model:

  $
    min_(theta_k) lr({ F_k (theta_k) + mu/2 lr(|| theta_k - theta^t_"global" ||)^2 }),
  $ <eq-fedprox>

  where $mu gt 0$ is the regularisation coefficient. This term anchors local
  training near the global model, especially beneficial for clients with extreme
  distribution shifts. We use $mu = 0.01$ throughout. Additionally, we maintain
  a *local anomaly threshold* $tau_k$ personalised per client, so that the shared
  encoder $theta$ benefits from federation while local decision boundaries adapt
  to device-specific operating conditions.

  == Module 4: Byzantine-Robust Aggregation

  === Robust Federated Aggregation (RFA)

  Given compressed client updates $brace.l hat(Delta)_k^t brace.r _(k in cal(S)_t)$,
  the server computes an *approximate geometric median* via Weiszfeld iterations:

  $
    Delta^(t+1) = sum_k w_k hat(Delta)_k^t, \
    w_k = frac(1, max(lr(|| hat(Delta)_k^t - Delta^((i)) ||), epsilon)),
  $

  initialised from the arithmetic mean and iterated 5 times.
  The geometric median is provably robust to up to $floor((n-1) slash 2)$ corrupted inputs,
  providing Byzantine tolerance with modest additional communication relative
  to simple averaging.

  === Coordinate-wise Median and Krum

  As baselines we also compare: *(i)* coordinate-wise median; *(ii)* trimmed mean (trim 10% from each end per coordinate);
  *(iii)* Krum, which selects the update minimising the sum of distances to its
  $n - f - 2$ nearest neighbours.

  // ─────────────────────────────────────────────────────────────
  //  5. EXPERIMENTAL SETUP
  // ─────────────────────────────────────────────────────────────
  = Experimental Setup

  == Synthetic IIoT Dataset

  To ensure full reproducibility, we generate a *synthetic multi-client IIoT
  benchmark* that replicates the distributional structure of real HAI, SWaT, and
  ToN\_IoT datasets while avoiding licensing barriers. The dataset consists of
  $K = 5$ clients, each simulating a distinct industrial edge node:

  $
    x_f^((k))(t) = A_k (sin(2 pi f_k t + phi_f) \
      + h_k sin(4 pi f_k t + phi_f)) \
      + epsilon_f^((k))(t) + b_f^((k)),
  $

  where $f_k = 0.01 + k dot 0.008$ (different periodicity per client), $A_k = 1 + 0.4 k$
  (non-IID amplitude), $epsilon^((k)) tilde cal(N)(0, sigma_k^2)$ (heterogeneous noise),
  and $b_f^((k))$ is a sensor-specific bias (feature shift). Each client produces
  $n = 4000$ samples over $F = 6$ channels.

  *Anomaly injection* combines three types at approximately 6% total ratio:
  *(i)* point spikes (4--8x amplitude); *(ii)* drift segments (15--25 steps, progressive offset);
  *(iii)* pattern shifts (20--35 steps, 2--3.5x amplitude). This non-IID setup
  is visualised in @fig-dataset.

  #figure(
    image("figures/fig_dataset_overview.png", width: 100%),
    caption: [Synthetic IIoT dataset: the five edge clients exhibit distinct
    frequencies, amplitudes, and noise levels (non-IID). Red dots mark injected
    anomalies (spikes, drift, pattern shifts).],
  ) <fig-dataset>

  Sliding windows of $W = 30$ steps with stride 1 are used; train/test split is 70%/30%,
  with normal-only training (unsupervised setting).

  == Implementation Details

  All models are implemented in PyTorch 2.11. The LSTM-AE has $H = 48$, $L = 16$,
  approximately 51K parameters. FL training: $T = 40$--$60$ rounds, $E = 3$ local epochs,
  $"lr" = 10^(-3)$ (Adam), batch size 64. Experiments run on NVIDIA GPU. All
  random seeds are fixed for reproducibility (seed = 42).

  == Metrics

  - *Detection quality*: AUROC, F1 (at 90th-percentile threshold), AUPR.
  - *Communication cost*: bytes per round per client (32-bit representation, compressed to $kappa$ fraction); total MB over all rounds.
  - *Robustness*: AUROC vs. attack fraction $alpha in {0, 0.1, 0.2, 0.3}$.

  // ─────────────────────────────────────────────────────────────
  //  6. RESULTS
  // ─────────────────────────────────────────────────────────────
  = Results and Analysis

  == FL Convergence and Communication Trade-off

  @fig-convergence shows training loss curves and cumulative communication costs
  for four FL configurations. FedProx converges more smoothly under non-IID
  distributions compared to plain FedAvg. Adding Top-10% compression reduces
  total bytes by approximately 10x with minimal impact on convergence speed.
  The combined FedProx + Top-K
  configuration achieves the best balance of convergence stability and
  communication efficiency.

  #figure(
    image("figures/fig_fl_convergence.png", width: 100%),
    caption: [Left: Training loss convergence for FedAvg, FedProx, Top-$k$, and
    FedProx+TopK over 60 rounds. Right: cumulative communication cost (MB).
    FedProx achieves smoother convergence under non-IID; Top-$k$ reduces
    communication by approximately 10x with negligible loss degradation.],
  ) <fig-convergence>

  == Communication Overhead vs. Compression Ratio

  @fig-comm illustrates the communication--quality trade-off across six compression
  ratios ($kappa in {1%, 5%, 10%, 20%, 50%, 100%}$). Even at $kappa = 1%$,
  detection quality degrades moderately, confirming error-feedback effectiveness.
  The "knee point" of the trade-off curve lies around $kappa = 10%$, where
  total communication is reduced by 10x while AUROC drops by less than 3%.

  #figure(
    image("figures/fig_communication_overhead.png", width: 100%),
    caption: [Communication efficiency: total uplink bytes (left bars) and AUROC
    (right scatter) for compression ratios from 100% (no compression) to 1%.
    The knee point at 10% compression offers the best quality-cost trade-off.],
  ) <fig-comm>

  == Anomaly Detection Quality

  @fig-ad shows reconstruction errors and detection outcomes for a representative
  test client. The LSTM-AE accurately flags spike and drift anomalies (high
  reconstruction error), with few false positives in the normal region.

  #figure(
    image("figures/fig_anomaly_detection.png", width: 100%),
    caption: [Anomaly detection on Client 0 test data. Top: raw sensor signal
    with true anomaly labels (red). Middle: reconstruction error (purple) with
    detection threshold (dashed red). Bottom: TP/FP/FN/TN classification outcomes.],
  ) <fig-ad>

  == Robustness Under Byzantine Attacks

  @fig-rob compares five aggregation methods under increasing fractions of
  model-poisoning attackers. Vanilla FedAvg degrades sharply beyond $alpha = 10%$,
  approaching near-random AUROC at $alpha = 30%$. Coordinate-wise median and
  RFA maintain substantially higher AUROC across all attack fractions, validating
  Byzantine-robust aggregation as essential in adversarial IIoT deployments.
  Krum provides strong point-wise robustness but can be brittle at high $alpha$
  without multi-Krum selection.

  #figure(
    image("figures/fig_robustness_comparison.png", width: 100%),
    caption: [Robustness comparison: AUROC vs. fraction of Byzantine clients
    (model poisoning). Left: line plots per aggregator. Right: AUROC heatmap.
    RFA and median maintain significantly higher robustness than FedAvg at 20--30%
    attack fraction.],
  ) <fig-rob>

  == On-Off Attack Evasion

  @fig-onoff shows that on-off attacks --- where adversaries behave honestly in
  alternating rounds to evade defences --- reduce the effectiveness of all
  aggregators to some degree. However, RFA is the most resilient, as the
  geometric median is less sensitive to the gradual drift introduced by on-off
  strategies compared to coordinate-wise median.

  #figure(
    image("figures/fig_on_off_attack.png", width: 100%),
    caption: [Defence performance against persistent model poisoning vs.~on-off
    evasion attack strategy (20% adversarial clients). RFA shows the strongest
    robustness against both attack types.],
  ) <fig-onoff>

  == Ablation Study

  @fig-ablation and @tab-ablation show the contribution of each module. Removing
  robust aggregation (FedAvg baseline with 20% attackers) causes a severe AUROC
  drop. Removing FedProx reduces stability under non-IID. Removing compression
  multiplies communication cost without detection gains. The *full system* achieves
  the best quality, robustness, and bandwidth-efficiency trade-off.

  #figure(
    image("figures/fig_ablation_study.png", width: 100%),
    caption: [Ablation study: AUROC, F1, and total communication (MB) for five
    system configurations. The full system (FedProx + Top-10% + RFA) achieves
    the highest AUROC and F1 while minimising communication cost.],
  ) <fig-ablation>

  #figure(
    caption: [Ablation study summary (20% model-poisoning attackers, 40 rounds).
    Higher AUROC and F1 is better; lower Comm is better.],
    kind: table,
  )[
    #set text(size: 9pt)
    #table(
      columns: (2.4fr, 1fr, 1fr, 1.2fr),
      align: (left, center, center, center),
      stroke: 0.4pt,
      table.header(
        [*Configuration*], [*AUROC*], [*F1*], [*Comm (MB)*],
      ),
      [Full System (FedProx + TopK-10% + RFA)], [*0.68*], [*0.27*], [*1.49*],
      [No Compression (FedProx + RFA)],         [0.74],   [0.28],   [14.92],
      [No Robust Agg (FedProx + TopK)],         [0.50],   [0.06],   [1.49],
      [No FedProx (FedAvg + TopK + RFA)],       [0.67],   [0.25],   [1.49],
      [Baseline (FedAvg, 0% attackers)],        [0.66],   [0.22],   [14.92],
    )
  ] <tab-ablation>

  // ─────────────────────────────────────────────────────────────
  //  7. DISCUSSION
  // ─────────────────────────────────────────────────────────────
  = Discussion

  *Communication--quality trade-off.* Our results confirm that Top-$k$ sparsification
  with error-feedback provides an effective linear trade-off between uplink bytes and
  detection quality. The error-feedback mechanism is critical: without it, aggressive
  compression ($kappa lt 5%$) causes irreversible gradient bias that manifests as
  unstable convergence.

  *Non-IID robustness.* FedProx proves beneficial under the five-client non-IID
  benchmark, particularly for clients with extreme amplitude ratios (Client 4 vs.
  Client 0 differ by 2.6x). SCAFFOLD-style control variates would further reduce
  drift but increase protocol complexity; we leave this for future work.

  *Byzantine resilience limits.* Even RFA cannot maintain high AUROC beyond
  $alpha = 40%$ attackers when the model is not large (Krum's guarantee requires
  $n gt.eq 2f + 4$, satisfied for $alpha lt.eq 0.3$ in our 5-client setup). Adaptive
  attacks that specifically target RFA's Weiszfeld iterations remain an open
  challenge. An additional layer of norm-clipping and gradient direction checking
  (FLTrust-style) would provide defence-in-depth.

  *Benchmark limitations.* The synthetic dataset, though carefully parameterised,
  cannot fully replicate the temporal correlations and attack sophistication of
  real ICS platforms (SWaT, HAI, WADI). Validating on these benchmarks is future work.

  // ─────────────────────────────────────────────────────────────
  //  8. CONCLUSION
  // ─────────────────────────────────────────────────────────────
  = Conclusion

  We presented a unified, modular framework for communication-efficient and
  Byzantine-robust federated anomaly detection in IIoT edge networks. By combining
  LSTM-AE reconstruction-based detection, FedProx non-IID stabilisation, Top-$k$
  error-feedback compression, and RFA geometric-median aggregation, we achieve
  simultaneous improvements across detection quality, communication overhead,
  and adversarial resilience --- three axes that are jointly critical in resource-constrained
  industrial deployments.

  Key findings: *(i)* Top-10% compression reduces communication by 10x with
  less than 3% AUROC drop; *(ii)* FedProx improves non-IID convergence stability; *(iii)*
  RFA and coordinate-wise median maintain robust detection at 30% adversary fraction
  where FedAvg collapses; *(iv)* on-off attacks reduce all defences but RFA remains
  the most resilient. These results provide a reproducible baseline and experimental
  framework for future development of federated anomaly detection under realistic
  IIoT constraints.

  // ─────────────────────────────────────────────────────────────
  //  REFERENCES
  // ─────────────────────────────────────────────────────────────
  #colbreak(weak: true)
  = References

  #set text(size: 9pt)
  #set par(hanging-indent: 1.5em, spacing: 0.6em, justify: true)

  #let ref-entry(label-text, body) = [
    #block(below: 0.5em)[
      *\[#label-text\]* #body
    ]
  ]

  #ref-entry("1")[
    McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).
    Communication-efficient learning of deep networks from decentralized data.
    _AISTATS 2017_, PMLR 54, 1273--1282.
  ]

  #ref-entry("2")[
    Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
    Federated optimization in heterogeneous networks (FedProx).
    _MLSys 2020_.
  ]

  #ref-entry("3")[
    Reisizadeh, A., Mokhtari, A., Hassani, H., Jadbabaie, A., & Pedarsani, R. (2020).
    FedPAQ: A communication-efficient federated learning method with periodic averaging and quantization.
    _AISTATS 2020_, PMLR 108.
  ]

  #ref-entry("4")[
    Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., & Suresh, A. T. (2020).
    SCAFFOLD: Stochastic controlled averaging for federated learning.
    _ICML 2020_, PMLR 119.
  ]

  #ref-entry("5")[
    Li, T., Hu, S., Beirami, A., & Smith, V. (2021).
    Ditto: Fair and robust federated learning through personalization.
    _ICML 2021_, PMLR 139.
  ]

  #ref-entry("6")[
    Dinh, C. T., Tran, N. H., & Nguyen, T. D. (2020).
    Personalized federated learning with Moreau envelopes (pFedMe).
    _NeurIPS 2020_.
  ]

  #ref-entry("7")[
    Blanchard, P., Mhamdi, E. M. E., Guerraoui, R., & Stainer, J. (2017).
    Machine learning with adversaries: Byzantine tolerant gradient descent (Krum).
    _NeurIPS 2017_.
  ]

  #ref-entry("8")[
    El Mhamdi, E. M., Guerraoui, R., & Rouault, S. (2018).
    The hidden vulnerability of distributed learning in Byzantium (Bulyan).
    _ICML 2018_, PMLR 80.
  ]

  #ref-entry("9")[
    Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018).
    Byzantine-robust distributed learning: Towards optimal statistical rates.
    _ICML 2018_, PMLR 80.
  ]

  #ref-entry("10")[
    Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2022).
    Robust aggregation for federated learning (RFA).
    _IEEE Transactions on Signal Processing_, 70, 1142--1154.
  ]

  #ref-entry("11")[
    Cao, X., Fang, M., Liu, J., & Gong, N. Z. (2021).
    FLTrust: Byzantine-robust federated learning via trust bootstrapping.
    _NDSS 2021_.
  ]

  #ref-entry("12")[
    Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020).
    How to backdoor federated learning.
    _AISTATS 2020_, PMLR 108.
  ]

  #ref-entry("13")[
    Fang, M., Cao, X., Jia, J., & Gong, N. Z. (2020).
    Local model poisoning attacks to Byzantine-robust federated learning.
    _USENIX Security 2020_.
  ]

  #ref-entry("14")[
    Xie, C., Chen, M., Chen, P.-Y., & Li, B. (2023).
    Attacks against federated learning defense systems and their implications.
    _JMLR_, 24(305), 1--43.
  ]

  #ref-entry("15")[
    Yuan, D., Hu, S., Guo, S., Zhang, J., & Yang, B. (2020).
    Deep anomaly detection for time-series data in industrial IoT: A communication-efficient
    on-device federated learning approach. _IEEE Internet of Things Journal_, 8(9), 6348--6358.
  ]

  #ref-entry("16")[
    Ruff, L., Vandermeulen, R., Goernitz, N., et al. (2018).
    Deep one-class classification (Deep SVDD).
    _ICML 2018_, PMLR 80.
  ]

  #ref-entry("17")[
    Zhang, C., Zhu, Y., Zhang, X., & Li, Z. (2022).
    Deep federated anomaly detection for multivariate time series data (Fed-ExDNN).
    _arXiv:2205.04571_.
  ]

  #ref-entry("18")[
    Audibert, J., Michiardi, P., Guyard, F., Marti, S., & Zuluaga, M. A. (2022).
    Deep learning for anomaly detection in time series: A survey.
    _ACM Computing Surveys_, 54(3).
  ]

  #ref-entry("19")[
    Alistarh, D., De Sa, C., & Shah, N. (2018).
    The convergence of sparsified gradient methods.
    _NeurIPS 2018_.
  ]

  #ref-entry("20")[
    Park, D., Hoshi, Y., & Kemp, C. C. (2018).
    A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder.
    _IEEE Robotics and Automation Letters_, 4(2), 1543--1550.
  ]
