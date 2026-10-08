# Paper preparation

`model-workbench.tex` is a standalone English systems-report draft using standard `article`, embedded references, and no external figures or bibliography files. It describes implemented boundaries and underlying principles; it does not claim new kernels, measured GPU speedups, universal support, or production certification.

Open the source in the built-in LaTeX editor for PDF preview. With a separately installed TeX distribution, compile twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error model-workbench.tex
pdflatex -interaction=nonstopmode -halt-on-error model-workbench.tex
```

Before submission:

1. Replace placeholder authors and affiliations with approved identities.
2. Add the actual public repository, license, release commit and archived experimental artifacts. No public URL is invented in this draft.
3. Run the documented hardware evaluation protocol, then add actual measured results if making performance claims. Synthetic fixtures are not those results.
4. Review all capability statements against the submission's exact revision; rerun automated tests and record command/output externally with the release.
5. Prepare a source archive containing the `.tex` file only for this standalone document, not auxiliary/log files or the whole application repository. If figures are added later, include their source dependencies.
6. Inspect arXiv's generated PDF and metadata during submission. Local compilation does not guarantee submission acceptance or endorsement.

arXiv does not require a unique paper template. This source uses a conventional LaTeX structure intended for its TeX processing workflow. Follow the current [official TeX submission instructions](https://info.arxiv.org/help/submit_tex.html). This work has not been submitted or accepted.
