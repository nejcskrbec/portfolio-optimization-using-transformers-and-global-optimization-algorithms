---
name: thesis-writing
description: >-
  House style for writing the Slovenian UL FRI master's thesis in
  `master-thesis-book/` (LaTeX). Use whenever drafting, editing, or reviewing any
  `.tex` in that book — chapters, captions, equations, algorithms, tables,
  examples, results/discussion prose. Codifies the LaTeX conventions, Slovene
  academic writing voice, explanation depth, worked-example format, and citation
  rules distilled from the author's bachelor thesis and the existing master book.
---

# Master's thesis writing style (UL FRI, Slovene)

This skill encodes the author's proven thesis style, distilled from the bachelor
thesis (`diploma-FRI-vzorec.tex`, topic: geometric 0/1 knapsack) and reconciled
with the conventions already in place in `master-thesis-book/`. When the two
differ, **the master book wins** — the goal is one consistent book.

Golden rule: read a neighbouring `master-thesis-book/chapters/*.tex` file before
writing, and match its density, idiom, and markup exactly. New prose must be
indistinguishable from what's already there.

---

## 1. Language & voice

- **Slovene throughout.** Formal academic register.
- **First-person plural ("mi" form)** for everything the author does or narrates:
  *obravnavamo, definiramo, bomo predstavili, smo implementirali, oglejmo si,
  vidimo, opazimo, sklepamo, izberemo, vstavimo*. Never "jaz"; never impersonal
  passive where the plural reads naturally.
- **Pedagogical, guiding tone.** Lead the reader by the hand: pose the question,
  then answer it (*"Kaj pa, če želimo kakovost rešitev še izboljšati?"*,
  *"V vsaki iteraciji si moramo postaviti dve vprašanji: …"*). Motivate before
  formalizing.
- **Honest, measured claims.** State limitations plainly (*"Tak pristop je
  seveda zelo časovno in prostorsko potraten"*, *"kar presega obseg tega
  dela"*). Prefer directional/economic language over overstated significance —
  do **not** claim statistical significance the data doesn't support (see the
  project's CLAUDE.md: ~46 monthly windows rarely reach p<0.05).
- Slovene typographic details: use `\,` inside numbers as the decimal comma via
  `1{,}24` in math/tables; abbreviations with a hard space `npr.\ `, `tj.\ `,
  `oz.\ `, `gl.\ `, `angl.\ `, `dr.\ `, `sod.\ `. Slovene quotes: ``\,…''`` or
  the book's `\enquote{…}` if defined; the bachelor used ``…''`.

## 2. Introducing terms and abbreviations

This is the single most recognizable habit — apply it consistently.

- **First occurrence of a technical term:** bold the Slovene term, immediately
  gloss the English in italics with `angl.`, then (if it will recur) define its
  acronym.
  ```latex
  \noindent\textbf{Postavitev} (angl.\ \emph{placement}) je preslikava, s katero …
  ```
- **Acronyms** are introduced once and reused: *"problem geometrijskega
  0/1-nahrbtnika (v nadaljevanju PGN) (angl.\ \emph{geometric knapsack
  problem})"*; thereafter always "PGN". The master book does the same for its
  scenarios (Zgodovinski, Transformer, Ansambel) and abbreviations.
- **English gloss uses `\emph{}`** (master-book convention), not `\textit{}`
  (bachelor). Convert any `\textit{}`-for-glosses you touch to `\emph{}`.
- Keep a running acronym list consistent with `front_pages` / *Seznam
  uporabljenih kratic*.

## 3. Document structure & chapter rhythm

Chapter files live in `master-thesis-book/chapters/NN-ime.tex`, wired in
`thesis_template.tex`. Current spine: `01-uvod`, `02-ozadje`,
`03-napovedni-model`, `04-optimizacijski-algoritmi`, `05-eksperimentalno`,
`06-razprava`, `07-sklep`.

- Start each chapter file with a comment banner and `\chapter{…}` + `\label{ch:…}`.
- **Uvod pattern** (proven in both theses): (1) situate the problem and its
  hardness, (2) real-world applications, (3) a *roadmap paragraph* in future
  tense — "Začeli bomo z…, nato bomo…, na koncu bomo…", (4) explicit main goal
  ("Glavni cilj … je bil …"), (5) a closing sentence of hoped contribution.
- **Method/algorithm sections follow a fixed template** (carry this over from the
  bachelor — it's the backbone):
  1. One-paragraph intro naming the method, its acronym, and a `\cite`.
  2. `\subsection{Definicije}` — the vocabulary the algorithm needs, each term as
     a bold `\noindent\textbf{Term} (angl.\ \emph{…})` definition, often with a
     figure.
  3. `\subsection{Izvajanje …}` — the pseudocode (`algorithm` env) followed by a
     prose walk-through and a numbered `enumerate` of the loop's steps.
  4. `\begin{primer}` — a worked example on a small running instance, with
     figures per iteration.
- Prefer `\paragraph{Naslov.}` mini-headings for conceptual beats inside a
  section (master-book style, e.g. `\paragraph{Nadzorovano učenje.}`) rather than
  over-splitting into many subsections.
- `\newpage` is used deliberately to keep figures/algorithms with their text —
  keep the author's manual page discipline; don't scatter `\clearpage`.

## 4. LaTeX markup conventions (canonical = master book)

- **Emphasis:** `\emph{}` for English glosses and for emphasized Slovene terms
  on later mention; `\textbf{}` for the defining first mention and for labels
  (`\noindent\textbf{Vhod:}`).
- **Math vectors/matrices:** `\vec{\mu}`, `\vec{X}`, `\vec{W}^Q` (bold via
  `\vec`, per ch. 03). Use `\mathbb{}` for number sets, `\top` for transpose,
  `\arg\min`, `\mathrm{}` for operators like `\mathrm{softmax}`.
- **Equations:** numbered `equation` with `\label{eq:…}`, referenced as
  `enačba~\eqref{eq:…}`. Multi-line: `align*` / `multline*` (unnumbered display
  is fine for derivations, as in the bachelor's definitions). Use `~` (non-break)
  before `\eqref`, `\ref`, and `\cite` anchors.
- **Figures:**
  ```latex
  \begin{figure}[htbp]   % or [H] to force position (float pkg is loaded)
    \centering
    \includegraphics[width=0.9\linewidth]{ime_slike}  % no extension, no path
    \caption{Poved z veliko začetnico, konča s piko.}
    \label{fig:ime}
  \end{figure}
  ```
  Images live in `master-thesis-book/figures/` (on `\graphicspath`), so reference
  by bare name. Side-by-side: two `\includegraphics[width=0.49\linewidth]{…}`
  split by `\hfill`, or `subcaption`'s `subfigure` for (a)/(b) panels.
- **Cross-references** always via `\ref`/`\eqref` with the Slovene noun and a
  non-breaking space: `sliki~\ref{fig:…}`, `algoritmu~\ref{alg:…}`,
  `tabeli~\ref{tab:…}`, `poglavju~\ref{ch:…}`, `primeru~\ref{pr:…}`. Never
  hard-code numbers. Label prefixes: `fig:`, `tab:`, `alg:`, `eq:`, `ch:`,
  `sec:`, `pr:` (examples).

## 5. Algorithms (pseudocode)

The style file loads `algorithm` + `algorithmic` and renames the float to
**Algoritem**, with `\algorithmicrequire`→`\textbf{Vhod:}` and
`\algorithmicensure`→`\textbf{Izhod:}`.

```latex
\begin{algorithm}[H]
\caption{Ime postopka}
\begin{algorithmic}[1]
\State \textbf{Vhod:} … opis vhoda …
\State \textbf{Izhod:} … opis izhoda …
\State \textit{spremenljivka} $\leftarrow$ začetna vrednost
\While{pogoj}
    \State … korak …
\EndWhile
\State \Return \textit{rezultat}
\end{algorithmic}
\label{alg:ime}
\end{algorithm}
```

- Comments/variable names in the pseudocode are Slovene and set with `\textit{}`.
- After every algorithm, add a prose paragraph that names the inputs/outputs and
  then a numbered `enumerate` explaining the loop body step by step (bachelor
  pattern — do not drop this; the pseudocode alone is never enough).
- Use `\Comment{…}` for inline notes, `\Function`/`\EndFunction` for helpers.

## 6. Worked examples — the `primer` environment

`primer` is a numbered theorem-like environment (`\newtheorem{primer}{Primer}[chapter]`).
This is the author's signature explanatory device.

```latex
\begin{primer}
\label{pr:ime}
Oglejmo si preprost primer izvajanja … za seznam … $I=\{1,2,3,4,5,6\}$ …
\begin{enumerate}
    \item …
\end{enumerate}
\end{primer}
```

- Reuse **one small running instance** across all examples in a chapter (the
  bachelor threaded `I={1,…,6}` through SNP, KHA, HSO, mutations). Consistency of
  the example makes methods directly comparable.
- Each example is a numbered, iteration-by-iteration trace, paired with figures
  that show the state at each step (`\ref{fig:…} (I)`, `(II)`, …).
- Open with an imperative invitation: *"Oglejmo si …"*, *"Denimo, da …"*.
- The master book also uses `primer` for conceptual illustrations (e.g. an
  attention-weight matrix), not only algorithm traces — both are in-scope.

## 7. Tables (canonical = booktabs)

Master book uses **booktabs**, not the bachelor's `\hline` grids. Prefer:

```latex
\begin{table}[htbp]
  \centering
  \caption{Opis tabele (nižje/višje je bolje, če je relevantno).}
  \label{tab:ime}
  \begin{tabular}{lcccc}
    \toprule
    Algoritem & povpr. & mediana & … \\
    \midrule
    PSO & 1{,}24 & 1{,}34 & … \\
    \bottomrule
  \end{tabular}
\end{table}
```

- Decimal comma via `1{,}24`. Caption **above** the table (as in ch. 05).
- Wide tables: `sidewaystable` (rotating) as the bachelor did for big result
  grids; keep numeric columns right/`c`-aligned and headers bold or in the
  first row above `\midrule`.

## 8. Scope & depth of explanation

Calibrate to the bachelor's demonstrated level — thorough but never padded:

- **Define before you use.** Every non-obvious term gets a bold definition (often
  with a figure) before it appears in an algorithm or proof.
- **Explain the "why", not just the "what".** After giving a formula or step,
  say what it buys you and what its failure mode is (e.g. why scaling attention
  by `\sqrt{d}`, why a fitness function can misbehave with a few large items).
- **One idea per paragraph.** Short-to-medium paragraphs; a new conceptual beat
  gets a new paragraph or a `\paragraph{}` head.
- **Match rigor to role.** Core contributions (the model, the objective, the
  ensemble) get full mathematical treatment; supporting background gets a
  compact, cited summary. Don't re-derive standard results — cite them.
- Keep the thesis framing straight (from CLAUDE.md): the claim is that the
  transformer *pipeline* (model μ + model Σ) *improves on* the classical
  historical pipeline via combination — "improves on", not "replaces"; the
  ensemble is a robustness layer, not the headline.

## 9. Results & discussion prose

Model on the bachelor's *Rezultati* / *Zaključek* and the master's
`05-eksperimentalno` / `06-razprava`:

- Open results with a **reproducibility + environment note**: where the code
  lives (`\cite{github}`), which script reproduces it, the caveat that
  stochastic methods vary run-to-run, language/libraries, and hardware.
- **Per-method commentary:** for each method/scenario, discuss quality *and*
  cost (time), compare against the others, and explain *why* it wins or loses in
  mechanistic terms — not just which number is bigger.
- Reference every table and figure explicitly and interpret them; never drop a
  float without prose that reads it.
- **Zaključek** recaps the journey (problem → methods → results), states the
  headline finding honestly, lists limitations, and proposes concrete future
  work.

## 10. Citations & bibliography

- **Cite every borrowed concept.** Every concept, term, method, model,
  metric or technique that we did *not* invent ourselves must carry a
  `\cite` to its source on first mention — e.g. the transformer, LSTM,
  MASTER, Adam, the information coefficient, factor risk models, the Hedge
  algorithm. Only the thesis's own contributions (e.g. MASTER-lite, our
  pipeline) go uncited. When in doubt, cite.
- **Master book uses BibTeX** (`\bibliographystyle{elsarticle-num}`,
  `\bibliography{bibliography}`) — numeric. Add entries to
  `master-thesis-book/bibliography.bib`. (The bachelor used biblatex/biber; do
  **not** reintroduce biblatex here.)
- Cite with `\cite{key}` attached to the term/method on first mention, with a
  non-breaking space: `MASTER~\cite{li2024master}`, `LSTM~\cite{hochreiter1997lstm}`.
- Attribute named methods to authors in prose when natural: *"ki so jo predlagali
  Li in sod.~\cite{li2024master}"*, *"tipologija, ki so jo definirali Wäscher,
  Haußner in Schumann~\cite{…}"*.
- Prefer `@article`/`@inproceedings` with `doi`/`url`; keep author lists complete.

## 11. Build

```bash
cd master-thesis-book && latexmk -pdf thesis_template.tex   # respects latexmkrc
```
The article-format companion paper is separate: `thesis-paper/main.tex`
(`latexmk -pdf`). Don't cross-wire their bibliographies or styles.

---

## Quick checklist before finishing any edit

- [ ] Slovene, "mi" voice, pedagogical framing.
- [ ] First mention of each term: `\textbf{term}` + `(angl.\ \emph{gloss})` + acronym.
- [ ] English glosses use `\emph{}`, not `\textit{}`.
- [ ] Every figure/table/algorithm/equation has a `\label` and is referenced with
      `~\ref`/`~\eqref` (no hard-coded numbers).
- [ ] New algorithm → Vhod/Izhod + prose walk-through + numbered steps.
- [ ] New method → Definicije → Izvajanje → Primer (on the running instance).
- [ ] Tables use booktabs + decimal comma; captions above.
- [ ] Claims are honest and directional; no overstated significance.
- [ ] Citations via BibTeX `\cite`; entry added to `bibliography.bib`.
- [ ] Reads indistinguishably from the neighbouring chapter.
