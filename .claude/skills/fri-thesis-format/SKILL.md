---
name: fri-thesis-format
description: >-
  Official UL FRI short-form thesis rules (the `friteza` journal template /
  `navodila-za-pisanje`) — structure, length, academic voice, Slovene typography,
  units, references, floats, code, math, and bibliography. Use whenever drafting,
  editing, or reviewing the short-form paper `thesis-paper/main.tex` (the friteza
  article), or any friteza `.tex`. For the long-form book use `thesis-writing`.
---

# UL FRI short-form thesis rules (friteza)

Condensed from the professors' official template `thesis-paper/navodila-za-pisanje.tex`.
These are hard requirements of the FRI `style/friteza` class. When editing
`thesis-paper/main.tex`, apply them exactly; when unsure, open
`navodila-za-pisanje.tex` for the canonical example.

Golden rule: **LaTeX + a bibliography database are mandatory** (see §9 — the
template says BibLaTeX but actually ships natbib/BibTeX), PDF/A-2b is required
(verify with veraPDF before submission), and every claim must be verifiable
(own derivation/experiment, or a citation).

---

## 1. Structure & length (the *gradniki*)

The exact structure depends on the field — consult the advisor — but every thesis
must carry these components (*gradniki*), in this reading order. The core sections
may be renamed, split, merged, or dropped as the topic requires.

1. **Naslovna stran** — SLO + ENG title (concise yet informative enough to reveal
   the CS domain), author, advisor, optional co-advisor, year and place.
2. **Povzetek (SLO) + Abstract (ENG)** — both mandatory; short self-contained text
   (~250 words) summarizing motivation, methods, key results.
3. **Ključne besede / Keywords** — **4–6**, specific (technologies, algorithms,
   domains); used for indexing.
4. **Uvod** — state of the field, motivation, goals and the hypotheses to be
   tested; **ends with a roadmap of the sections** that uses `\ref` (never
   hard-coded section numbers), each heading carrying a `\label`.
5. **Pregled področja / sorodna dela** — position the work in context: which
   solutions already exist and how ours differs; a thorough literature review.
6. **Osrednji razdelki** (core, flexible):
   - **Metodologija / teoretične osnove** — concepts needed to understand the solution.
   - **Analiza in načrtovanje** — architecture, constraints, requirements.
   - **Implementacija** — the solution, development details, libraries, technical
     challenges (no result numbers here).
   - **Eksperimenti, evalvacija in rezultati** — testing, comparisons/user study,
     results and argumentation (often split into several sections).
7. **Zaključek** — achievements + critical appraisal; **state limitations** and
   sketch future work.
8. **Literatura in viri** — every listed source is cited in the text and vice versa.
9. **Priloge** (optional) — bulky material that would disrupt the main flow
   (long code, questionnaires, detailed diagrams).

**Length** (incl. references, excl. appendices): diploma **8–12 pp**, master's
**11–20 pp**. **Abstracts**: 150–250 words each. **Long Slovene abstract**
(`longsloveneabstract`, only when the thesis is written in English): ~**10 %** of
body length.

Reading order ≠ writing order: write related work and methodology before
experiments, and write the **intro and abstract last** (don't pretend the
experiments aren't done yet).

## 2. Academic voice (Slovene)

- **First-person plural**: *razvili smo, opazimo, ugotovimo*. Passive is allowed
  but clunky (*razvito je bilo*). **Never** first-person singular except to mark a
  personal design choice among alternatives.
- Short, unambiguous sentences; consistent terminology and notation for each
  concept throughout; define new terms precisely at first use.
- Use established Slovene technical terms; find/coin Slovene equivalents (with the
  advisor) for terms lacking them.
- Prefer Slovene over loanwords: **samodejno** (not avtomatsko), **odstotek** (not
  procent), **poskus** (not eksperiment), **težava** (not problem). But avoid
  forced/false translations (*konkurenčnost* ≠ *concurrency* → *sočasnost*).

## 3. Slovene typography (the frequent mistakes)

- **Word order — non-declined modifiers (acronyms/brand names) go AFTER the noun**:
  *poizvedba SQL* (not *SQL poizvedba*), *vmesnik API*, *strežnik SQL*,
  *protokol TCP/IP*, *os $x$*, *procesorji Intel* / *Intelovi procesorji*. Never
  *SQL poizvedba*, *Intel procesor*.
- **Declining foreign names — no hyphens**: *Applov*, *Skypov* (not *Apple-ov*).
- **Dashes** (critical, most common source of errors):
  - **Vezaj `-`** (hyphen, sticky): equal-rank compounds — *slovensko-angleški*.
  - **Pomišljaj `--`** (en-dash): in Slovene it is **non-sticky — with spaces on
    both sides**, used for parenthetical breaks. **This is the parenthetical dash
    to use in Slovene prose.**
  - **Razponski `--`** (range, sticky): *strani 7--11*, *od--do*.
  - **Em-dash `---`**: **NEVER in Slovene.** (It is the English sticky dash,
    *word---word*.) Do not use `---` anywhere in a Slovene thesis; use spaced `--`.
- **Quotes**: use `\enquote{...}` (produces »Slovene« quotes), **never** literal
  `"`, `''`, `` `` ``, or literal `»«` characters.
- **Ellipsis**: do **not** use `...` in a thesis; write **itd.** or **idr.** In one
  sentence use only one of *npr.* / *itd.* (not both). (If ever needed elsewhere,
  `\dots`, not three periods.)
- **Sticky punctuation**: `.` `!` `?` are left-sticky (no space before). Avoid `!`
  and `?` in technical text. Parens: `(` right-sticky, `)` left-sticky.
- **Non-breaking space `~`** before every `\ref`, `\eqref`, `\cite`
  (*Slika~\ref{...}*, *enačba~\eqref{...}*, *...~\cite{...}*) and inside
  abbreviations that must not break: *prof.~dr.~Novak*.
- **French-spacing fixes**: mid-sentence abbreviation → *npr.\ v tem primeru*
  (backslash-space) so LaTeX doesn't insert an end-of-sentence gap; sentence
  ending in an uppercase letter/acronym → *...CPE\@. Sledi...* (`\@` before the
  period).
- **Commas**: predicates separated by commas **except** when joined by
  *in, pa, ter, ne–ne, niti–niti, ali, bodisi, oziroma*. Watch multi-word
  conjunctions (*tako da*, *kljub temu da*) and the *kot* trap
  (*boljše kot prejšnja leta* vs *boljše, kot so bile*). Prefer short clauses.
- Capitalize only the **first** word of multi-word proper names/program names
  (*Telefonski imenik Slovenije*), except genuine foreign product names
  (*Microsoft Office*).

## 4. Quantities & units (`siunitx`)

- Quantity symbols in **italic** ($f$, $B$, $C$); unit symbols **upright**
  (\unit).
- Value + unit with `\qty{100}{\giga\byte}`; a tilde (non-breaking, sticky space)
  always sits between number and unit.
- **Decimal comma** in Slovene: `output-decimal-marker={,}` globally; write
  numbers as $0{,}97$.
- Percent/degrees with a thin space: `\qty{98}{\percent}` → 98 %, `\qty{85}{\degreeCelsius}`.
- Respect **SI**; decimal prefixes (k, M, G, T = ×1000) vs binary (Ki, Mi, Gi, Ti
  = ×1024); distinguish bit \unit{\bit} from byte \unit{\byte}.

## 5. References & citing

- `\label{...}` with prefixes: `fig:`, `tab:`, `sec:`, `eq:`, `alg:`, `code:`.
- `\ref` for sections/floats, **`\eqref`** for equations (adds parentheses;
  never `\ref` an equation), `\cite` for sources — each preceded by `~`.
- When *naming* a float, capitalize: *glej Sliko~\ref{...}*, *Algoritem~\ref{...}*,
  *Tabela~\ref{...}*. `.` after `\cite{}`, not before.
- Autonumber the roadmap; never write section numbers by hand.

## 6. Floats (figures & tables)

- Refer to **every** float in the text; keep the reference near the float.
- **Figure captions BELOW**, **table captions ABOVE**.
- Graphs/diagrams as **vector `.pdf`** (e.g. save matplotlib as pdf), never
  `.png`/`.jpg`; photos may be raster. Captions should be self-explanatory.
- Multiple floats: `subcaption` (`subfigure`/`subtable`) — **not** `subfig`,
  `subfigure`(old), or TikZ for layout. Wide floats: `largefigure`/`largetable`.

## 7. Code & algorithms

- Source code: `minted`. Pseudocode: `algorithm` + `algpseudocode`, `[1]` for line
  numbers, **caption on top**, treated as a float and `\ref`-ed.

## 8. Math

- Equations are part of the sentence → they carry punctuation (comma mid-sentence,
  period if the sentence ends there).
- Upright standard functions: `\sin`, `\cos`, `\ln`, `\max` (not *sin*).
- `align` for aligned derivations; `split` for one long numbered equation; `cases`
  for piecewise; `amsmath` `bmatrix`/`pmatrix` for matrices; `\dots`/`\vdots`/`\ddots`.

## 9. Bibliography

- The prose of `navodila-za-pisanje.tex` calls **BibLaTeX** mandatory
  (`style=numeric`, `sorting=nty`), but the shipped class contradicts it:
  `style/friteza.cls:211` loads **natbib** and line 214 sets
  `\bibliographystyle{style/pnas2011}`, and the template's own document body calls
  `\bibliography{...}`. **In practice this is BibTeX** — use `\bibliography{}` and
  run `bibtex`; do not add `\addbibresource`/`\printbibliography`, they will not
  compile. One `.bib` file; symbolic keys.
- Only cited sources appear; every listed source is cited. Prefer peer-reviewed,
  stable sources; include **DOI**; for web sources give an access date.
- **Verify autogenerated `.bib` entries** (wrong entry type is common; LLM-invented
  references are a real risk).

## 10. Pre-submission checklist

- Correct element usage (dashes, abbreviation spacing, `\enquote`, `~` before
  refs/cites); grammar; no text pasted verbatim from an LLM (read/verify all).
- Enough figures; every figure/table both captioned **and** referenced.
- Correct study programme, author, advisor/co-advisor names.
- No overlong lines (compile with `draft` to spot them); vector figures included;
  no undefined citations (search the PDF for `?`).
- PDF/A compliance (veraPDF).
