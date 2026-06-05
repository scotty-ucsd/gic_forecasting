# Final Milestone Submission Guide

This document organizes the final milestone instructions into a clean, report-ready checklist and structure reference.

## Submission Overview

For the final milestone, submit:

- A report as a PDF file.
- Code as a ZIP file.
- Only **one member** of the group needs to submit on behalf of the team.

## Required Deliverables

### 1. Report

- Submit the final report in **PDF** format.
- Use the naming convention: `<Group number>-Report.pdf`
- Example for Group 42: `42-Report.pdf`

### 2. Software Code

- Submit a ZIP archive containing all source code.
- Use the naming convention: `<Group number>-SW.zip`
- Example for Group 42: `42-SW.zip`
- The ZIP file may include code in formats such as `*.py`, `*.ipynb`, or `*.m`.
- Include a `README.txt` with instructions explaining how to compile and run the code.

## Report Expectations

### General Notes

- This is **not a rubric**.
- Including every listed section does **not** guarantee a specific grade.
- The structure is provided to help organize the report.
- The report should read like an academic paper.
- Include the **group number in the title**.

### Length

There is some inconsistency in the posted instructions, so use the stricter interpretation unless your instructor clarified otherwise:

- One part says the report should be **8–10 double-column pages**, excluding contributions, reflections, and references.
- Another part says the report content should be **strictly 6 pages**, including figures, but excluding references, contribution details, and responses to critique.
- The conclusion section also says: “Report should be 6 pages to this point.”

### Recommended Safe Interpretation

- Treat **6 pages of main report content** as the safest target.
- Exclude the following from the page limit unless told otherwise:
  - References/Bibliography
  - Contributions
  - Reply to review / responses to critique
- Keep reflections separate as instructed.
- If needed, confirm the page limit with the course staff.

## Suggested Report Structure

## Title Page Details

- Include the **group number on the first page**, above the project title.
- Include the project title.

## Abstract

**Target length:** about 1 paragraph

Include:

- Motivation for the project.
- A high-level summary of the methodology.
- A concise summary of the main results.

## Introduction

**Target length:** no more than 1 page

Include:

- The problem being addressed.
- Why the problem matters.
- Motivation for choosing the problem.
- Any necessary background.
- A clear statement of the input and output of the system.

Example framing:

> “The input to our algorithm is an {image, amplitude, patient age, grayscale video}. We then use a {SVM, neural network, linear regression, etc.} to output a predicted {age, stock price, cancer type, music genre, etc.}.”

This input/output description is especially important because teams may work in very different application domains.

## Related Work

**Target length:** no more than 1 page

Include:

- Existing papers relevant to the problem.
- Group related papers by approach or methodology.
- Strengths and weaknesses of prior methods.
- How previous work is similar to and different from your work.
- Which approaches seem especially strong or clever.
- The current state of the art.
- Whether the task is commonly done by hand or algorithmically.

Minimum expectation:

- Aim for **at least 5 references** in the related work section.

Suggested source:

- [Google Scholar](https://scholar.google.com/)

Also include:

- Previous attempts at the same problem.
- Prior technical methods.
- Previous learning algorithms used for similar tasks.

## Dataset and Features

**Target length:** no more than 1 page

Include:

- Description of the dataset.
- Number of training, validation, and test examples.
- Preprocessing steps.
- Normalization and/or data augmentation, if used.
- Data characteristics such as image resolution or time-series discretization.
- A citation for where the data was obtained.

Also discuss:

- What predictive tasks are possible with the dataset.
- Why your team selected one task over others.

Examples of possible tasks:

- Object presence classification.
- Localization.
- Breed/type/category classification.
- Time-series prediction.
- Regression or classification depending on labels.

Feature discussion should include:

- Raw features used.
- Engineered features.
- Extracted features such as Fourier transforms, PCA, ICA, etc.
- Example data samples in the report, such as images, waveforms, or representative records.

## Methods

**Target length:** about 2 pages

Include:

- A description of each learning algorithm or proposed method.
- Relevant mathematical notation.
- Important formulas where appropriate.
- A short explanatory paragraph for each algorithm.

The goal is to demonstrate understanding of how the algorithms work, not just list them.

Examples of content to include:

- Optimization objective for SVM.
- Softmax definition.
- Loss function.
- Model architecture.
- Training procedure.
- Any niche or cutting-edge algorithm details if relevant.

## Experiments, Results, and Discussion

**Target length:** about 2 pages

Include:

- Hyperparameters used.
- Why those hyperparameters were chosen.
- Whether cross-validation was used and how many folds.
- Primary evaluation metrics, explained before presenting results.
- Metric equations if needed.

Results should include a mix of:

- Tables.
- Plots.
- Visualizations.
- Error or failure-case analysis.

For classification tasks, include:

- Confusion matrix.
- Precision.
- Recall.
- Accuracy.

For regression tasks, include:

- Average error and other relevant regression metrics.

Discussion should address:

- Why certain algorithms succeeded or failed.
- What the figures and tables show.
- Failure cases and likely causes.

### Figure Requirements

All plots and figures should:

- Have legends where applicable.
- Have axis labels.
- Use font sizes that remain readable after being placed into the final report.
- Be discussed explicitly in the main text.

## Conclusion

**Target length:** about 1 paragraph

Include:

- A summary of the overall report.
- The key findings.
- Which algorithms performed best.
- Why some approaches may have outperformed others.

Also include:

- A **gap analysis** paragraph explaining:
  - The original goals.
  - What was ultimately accomplished.
  - Why there were any differences between goals and outcomes.

## Contributions

**No page limit**

Include:

- A short paragraph describing each author’s contribution.

Important note:

- This is a group report, and there is no extra credit for one person doing more work.
- If a team member does not contribute, the instructions say to email the professor early so the issue can be corrected.
- Non-participation may be penalized.

## Reflections

**Target length:** about half a page

Include:

- What the team would do differently if repeating the project.
- Whether different goals would have been better.
- Whether a different dataset would have been better.
- Lessons learned from the project process.

## References / Bibliography

**No page limit**

Include citations for:

1. Papers mentioned in the related work section.
2. Papers describing the algorithms used.
3. Codebases, software, and libraries used.

Examples of libraries that should be cited if used:

- scikit-learn
- TensorFlow
- Matlab toolboxes

The references section is excluded from the page limit to support a stronger literature review.

## Reply to Review

**Target length:** no more than 1 page

Include:

- A short, action-oriented reply to peer reviews.
- Responses should be concise and specific.

### Review File Naming Example

If your team is **Group 42** and receives feedback from **Teams 1, 13, and 47**, the review filenames would be:

- `reviewG1_42`
- `reviewG13_42`
- `reviewG47_42`

## Formatting Requirements

Use:

- **IEEE double-column format**
- IEEE templates: [IEEE conference templates](https://www.ieee.org/conferences/publishing/templates.html)

Available template types include:

- LaTeX
- Word (`.doc`)

### Referencing Expectations

- All figures not produced by the group should include a reference, such as `[3]`.
- Each time a new topic is introduced, a reference introducing that topic is expected.

## Common Mistakes to Avoid

The instructions specifically warn against these common issues:

1. **Missing group number on the first page**  
   - Include the group number above the project title without fail.

2. **Violating the page limit**  
   - Strictly follow the main report page limit.
   - The page limit includes figures.
   - The page limit excludes references, contribution details, and responses to critique.

3. **Unreadable figures**  
   - Ensure all figures include legends and labels.
   - Make sure fonts remain legible after figures are resized in LaTeX or Word.

## Rubric Notes

The instructions state: **See the rubric for details.**

### Visible Rubric Items

| Criterion | Description | Points |
|---|---|---:|
| Report | Evaluated based on how comprehensive the work is, not just the writing quality; also assesses how well the project is communicated to an audience unfamiliar with it. | 12 |
| Code | Evaluated on documentation quality, codebase complexity, number/complexity of models, and how well organized the code is in Git. | 25 |
| Reproducibility of results | The staff will attempt to reproduce the reported test results based on the provided instructions. | Not shown in provided excerpt |

## Final Submission Checklist

Before submitting, verify all of the following:

- Group number appears on the first page above the title.
- Report is in PDF format.
- Code is in a ZIP archive.
- PDF filename follows `<Group number>-Report.pdf`.
- ZIP filename follows `<Group number>-SW.zip`.
- ZIP archive includes all source code.
- ZIP archive includes `README.txt` with run/compile instructions.
- Report uses IEEE double-column formatting.
- Figures have readable labels, legends, and axis text.
- References are included for external figures, papers, algorithms, and libraries.
- Main report respects the page limit.
- Contributions section is included.
- Reflections section is included.
- Reply to review is included.

## Final Step

Once all files are ready:

- Upload the final report.
- Upload the software ZIP file.
- Press the **Submit** button in the Canvas assignment.
