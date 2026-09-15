# One-shot structural restructure of docs/paper/main.tex (advisor meeting fixes):
# 1) new ~180-word abstract (replaces old + duplicated tail)
# 2) merge "Corruption Framework" as subsection 4.1 of the metrics section
# 3) move the XAI section into Discussion (demote headers)
# 4) demote Conclusion to a Discussion subsection + expand to ~185 words
# 5) update the intro roadmap sentence
import sys

p = 'docs/paper/main.tex'
s = open(p, encoding='utf-8').read()
BS = chr(92)  # backslash


def lit(x: str) -> str:
    """Convert '\\section{...}' style markers written with single backslashes
    in this source file into the real LaTeX text containing backslashes."""
    return x.replace('%BS%', BS)


# markers (written with %BS% placeholder to avoid escaping issues)
M_ABS_B = lit('%BS%begin{abstract}')
M_ABS_E = lit('%BS%end{abstract}')
M_SEC_XAI = lit('%BS%section{XAI Analysis}')
M_SEC_DISC = lit('%BS%section{Discussion}')
M_SEC_CONC = lit('%BS%section{Conclusion}')
M_SEC_DATA = lit('%BS%section*{Data and code availability}')
M_SEC_FRAME = lit('%BS%section{Corruption Framework}' + '\n' + '%BS%label{sec:framework}')
M_SEC_METRICS = lit('%BS%section{Evaluation Metrics and Robustness Score}' + '\n'
                    + '%BS%label{sec:metrics}' + '\n\n' + 'For every (dataset type')
M_PRACTICE = lit('%BS%textbf{What the results mean for practice.}')

new_abs = lit(r'''%BS%begin{abstract}
Deep learning models detect Parkinson's disease (PD) from hand-drawn spiral,
wave, and meander images with high reported accuracy, yet these figures are
obtained on clean data, while deployed models face compression, blur, noise,
and imperfect acquisition. We present PD-RobustBench, a reproducible
framework that evaluates PD-detection models under eight acquisition-inspired
corruptions at five severity levels, following the ImageNet-C methodology,
and additionally measures how the models' Grad-CAM explanations change under
the same corruptions. Prediction and explanation degradation are combined
into a Parkinson Robustness Score (PRS, 0--100). Four ImageNet-pretrained
backbones (ResNet50, VGG16, EfficientNet-B0, MobileNetV3-Large) are adapted
with light head training on two public datasets (Parkinson's Drawings and
HandPD), yielding sixteen configurations that are evaluated across 41
conditions and three random seeds. Rotation is the most damaging corruption
(balanced-accuracy loss of 10--22PCTSIGN), whereas contrast is nearly harmless.
Grad-CAM maps decorrelate strongly at high severity (Spearman $%BS%rho=0.29$
under rotation), so explanations can drift while predictions survive. PRS
separates backbones (82--96) with explicitly reported seed and weight
sensitivity. The framework supports standardized, explainability-aware
robustness reporting and is intended for methodological benchmarking rather
than clinical validation.
%BS%end{abstract}''')

# ---- 1. abstract ----
a0 = s.index(M_ABS_B)
a1 = s.index(M_ABS_E) + len(M_ABS_E)
s = s[:a0] + new_abs + s[a1:]
s = s.replace('22PCTSIGN', BS + '%')  # escaped percent for LaTeX

# ---- 2. framework merge ----
new_frame = lit('''%BS%section{Evaluation Metrics and Robustness Score}
%BS%label{sec:metrics}

%BS%subsection{Corruption Framework}
%BS%label{sec:framework}''')
assert M_SEC_FRAME in s, 'framework header not found'
s = s.replace(M_SEC_FRAME, new_frame)
assert M_SEC_METRICS in s, 'metrics header not found'
s = s.replace(M_SEC_METRICS, lit('%BS%subsection{Evaluation metrics}\n\nFor every (dataset type'))

# ---- 3. XAI -> Discussion ----
i = s.index(M_SEC_XAI)
j = s.index(M_SEC_DISC)
xai = s[i:j]
s = s[:i] + s[j:]
xai = xai.replace(M_SEC_XAI, lit('%BS%subsection{XAI Analysis}'))
for h in ('Quantitative stability', 'Qualitative observations',
          'Stability is not validity'):
    xai = xai.replace(lit('%BS%subsection{' + h + '}'),
                      lit('%BS%subsubsection{' + h + '}'))
k = s.index(M_PRACTICE)
s = s[:k] + xai + '\n' + s[k:]

# ---- 4. conclusion demote + expand ----
i = s.index(M_SEC_CONC)
j = s.index(M_SEC_DATA)
new_concl = lit(r'''%BS%subsection{Conclusion}
%BS%label{sec:conclusion}
We presented PD-RobustBench, a reproducible corruption-robustness and
explainability evaluation framework for deep-learning detection of
Parkinson's disease from hand-drawn images. The benchmark covers two public
datasets and four image types, four ImageNet-pretrained backbones, eight
corruption families at five severity levels, a Grad-CAM stability analysis
with a randomization control, and a decomposable Parkinson Robustness Score
that was evaluated over three training seeds and three weighting schemes.
The benchmark supports three main conclusions. First, geometric misalignment
(rotation) and uneven illumination (shadows) degrade all evaluated backbones
far more than compression or sensor noise; both factors are largely
controllable at acquisition time. Second, explanation maps drift faster than
predictions under common compressions, so accuracy alone conceals a failure
mode that matters whenever saliency maps are shown to clinicians. Third,
backbone rankings depend on the corruption family and are only partially
stable across seeds, which argues for reporting full degradation profiles
with uncertainty rather than single numbers. All code, corruption
definitions, trained heads, per-image predictions, saliency maps and
sensitivity analyses are released, and new backbones can be added through a
three-point registration interface, so the benchmark can grow with the
field.

''')
s = s[:i] + new_concl + s[j:]

# ---- 5. intro roadmap ----
i = s.index('Section~' + BS + 'ref{sec:related} reviews related work')
anchor = 'findings, limitations, and conclusions.'
j = s.index(anchor) + len(anchor)
s = s[:i] + ('Section~' + BS + 'ref{sec:related} reviews related work. '
             'Section~' + BS + 'ref{sec:data} describes the datasets and models. '
             'Section~' + BS + 'ref{sec:metrics} defines the corruption framework, '
             'the evaluation metrics and the PRS. Section~' + BS + 'ref{sec:results} '
             'presents the results, and Section~' + BS + 'ref{sec:discussion} '
             'discusses practical implications, XAI stability, limitations and '
             'conclusions.') + s[j:]

open(p, 'w', encoding='utf-8').write(s)
print('restructured OK')
