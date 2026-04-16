# SoftwareX Revision Report

## 1. Section Mapping Report

| Original manuscript section | SoftwareX revision target | Transformation |
|---|---|---|
| Introduction | 1. Motivation and significance | Compressed broad background and reframed the paper around the software artifact rather than a standalone FL method. |
| Method Positioning | 1. Motivation and significance | Reduced literature review and replaced it with a concise statement of the practical software gap. |
| Problem Setup | 2. Software description | Folded problem formulation into the software narrative and implementation context. |
| Integrated Method | 2. Software description | Rewritten as software overview, module interaction, architecture, and implementation details. |
| Experimental Setup | 2.4 Dependencies, operating environment, and reproducibility notes + 3. Illustrative examples | Split into reproducibility information and concise execution-oriented evidence. |
| Results and Interpretation | 3. Illustrative examples | Reframed from hypothesis-style evaluation into software behavior and usage evidence. |
| Discussion | 4. Impact + 5. Limitations | Separated practical reuse value from technical and packaging limitations. |
| Conclusion | 6. Conclusions and future work | Rewritten to emphasize the repository and reproducible workflow as the main contribution. |
| Code and Data Availability | Code metadata table + Data availability | Moved repository details out of running prose and into SoftwareX-style metadata and end matter. |
| Funding | Funding | Retained with confirmation note. |
| Acknowledgements | Acknowledgements | Retained with confirmation note. |
| Declaration of Competing Interest | Declaration of competing interest | Retained in SoftwareX wording. |
| References | References | Kept selectively and trimmed conceptually in the body. |

## 2. SoftwareX Compliance Checklist

| Item | Status | Notes |
|---|---|---|
| Title page | still missing author input | Title and author names are present, but full affiliation addresses and corresponding-author details required by the guide are still absent. |
| Abstract | revised to comply | Rewritten as a software abstract focused on implementation and reproducibility. |
| Keywords | revised to comply | Reduced to concise indexing terms. |
| Metadata table | revised to comply | Added C1-C9 table with placeholders where facts are missing. |
| Software-centered structure | revised to comply | Replaced research-style section flow with SoftwareX-style sections. |
| Word-count compression | revised to comply | Rewritten toward the 3000-word limit stated in the guide. Final count should be verified on plain manuscript text, excluding title, authors, affiliations, references, and metadata tables. |
| Maximum six figures | revised to comply | Main manuscript now references six figures. |
| Highlights file | revised to comply | Added `highlights.txt` with five short bullet points for submission. |
| Open-source repository | compliant | Public GitHub repository is present. |
| License | still missing author input | No top-level `LICENSE` file found. |
| Archived release / reproducible capsule | still missing author input | No Zenodo/DOI/tagged archival release found. |
| Data statement | revised to comply | Added synthetic-data availability statement. |
| Funding statement | revised to comply | Included with author confirmation note. |
| Competing interest statement | compliant | Included. |
| CRediT statement | revised to comply | Added draft statement with author confirmation required. |
| AI declaration | compliant for now | Removed from the manuscript body because the guide requires it only if AI tools were used and must be declared at submission when applicable. |
| Reproducibility notes | revised to comply | Added entry point, dependencies, repository structure, and limitations. |
| Template compliance | revised to comply | Rewritten in SoftwareX style, but final journal-template adaptation may still require author-side formatting adjustments. |
| Separate figure requirements | revised to comply | Final main-manuscript set reduced to six figures. |

## 3. Figure Reduction and Consolidation Plan

### Keep in main manuscript

1. `fig_architecture_diagram`
   Reason: software architecture and integrated workflow anchor.
2. `fig_fl_convergence`
   Reason: shows training behavior and cumulative communication in one integrated figure.
3. `fig_communication_overhead`
   Reason: best compact illustration of communication-quality trade-off.
4. `fig_anomaly_detection`
   Reason: makes the local inference output of the software visible.
5. `fig_robustness_comparison`
   Reason: compact robustness comparison across aggregators and attack fractions.
6. `fig_ablation_study`
   Reason: shows how the integrated pipeline components contribute differently.

### Move to supplementary material

1. `fig_dataset_overview`
   Reason: useful context, but not essential once the dataset is described textually in the software section.
2. `fig_on_off_attack`
   Reason: informative but secondary relative to the main robustness figure and ablation figure.

### Keep as tables, not figures

1. SoftwareX code metadata table
2. Core module table

### Final main-manuscript figure count

6 figures total.

## 4. Missing Repository or Submission Items

1. Top-level `LICENSE` file.
2. Tagged software release corresponding to the submitted version.
3. Permanent archive or DOI-backed release, such as Zenodo.
4. Support/contact email for `C9`.
5. Confirmed affiliation and corresponding-author metadata.
6. External developer documentation or confirmation that `README.md` is the only manual.
7. Final AI declaration, if required.
8. Final funding confirmation.
9. Final CRediT role confirmation.
10. Optional supplementary package for figures moved out of the main manuscript.
11. `LICENSE.txt` filename or equivalent accepted open-source license file to satisfy the guide explicitly.

## 5. Consistency Notes Applied in the Revision

1. The manuscript no longer claims a MethodsX article identity.
2. The software is positioned as the contribution, not a standalone algorithmic novelty claim.
3. Latent size is reported as 12 in the implemented configuration.
4. Thresholding language was rewritten conservatively:
   The codebase supports percentile thresholding utilities and local visualization, but the current aggregate evaluation script uses a pooled 90th-percentile threshold over combined scores.
5. Robust aggregation claims were softened so that median and RFA are described as stable options in the reported runs rather than assigning universal superiority to one method.

## 6. Guide-Specific Notes Applied

1. The revision now follows the SoftwareX guide rather than a generic software-paper structure.
2. The manuscript remains a short descriptive paper centered on the software distribution.
3. The guide's recommendation for a separate highlights file was implemented as `highlights.txt`.
4. The guide states that the AI declaration section should appear only when needed, so the visible placeholder section was removed from the manuscript.
5. The guide requires GitHub availability, a readable `README.md`, and a license file. The repository satisfies GitHub and README requirements but still lacks a visible top-level license file.
