# Nerfies-Style Project Website Redesign

## Objective

Replace the current bespoke project landing page with a conventional academic project page based directly on the Nerfies website template. Preserve all substantive content currently published, but present it with the language, hierarchy, and restraint of the paper.

This redesign applies only to the website under `docs/`. It does not change the manuscript, repository README, evaluation code, dataset card, experimental outputs, or reported results.

## Explicit Requirements

- Use the actual Nerfies/Bulma page structure as the visual and structural base.
- Retain all substantive topics, figures, qualifications, and numerical results from the current website.
- Use the final manuscript as the source of truth for prose and claims.
- Keep prose close to the paper while shortening sentences for web reading.
- Remove promotional, slogan-like, and AI-style language.
- Use short paper-like section headings.
- Show `EMNLP 2026` as a main-conference venue.
- Link to Code and Dataset. Do not show a Paper button until the arXiv page is live.
- Lead with the existing task-example figure.
- Add a new diagram that makes iterative retrieval explicit.
- Preserve the acknowledgments and DSTA funding statement.
- Credit the Nerfies template visibly in the footer.
- Preserve responsive behavior, accessibility, metadata, social metadata, and favicons.

## Non-Goals

- Do not alter paper claims or results.
- Do not add new experiments, numbers, datasets, or interpretations.
- Do not redesign the GitHub README or Hugging Face dataset card.
- Do not retain the current landing-page aesthetic merely for continuity.
- Do not add analytics, video, a carousel, a lightbox, or decorative animation.
- Do not add a paper download or external paper link before arXiv is available.

## Source of Truth

The final manuscript is authoritative for terminology, numerical values, qualifications, and claim strength. The existing website is a checklist of content and figures that must remain represented.

The editorial transformation is:

`final manuscript claim -> concise web prose with the same meaning and qualification`

The redesign must not derive claims from the current website when the manuscript uses different or more careful wording.

## Page Architecture

### 1. Navigation

Use a thin, non-floating navigation row. It must not resemble the current rounded floating header. The desktop links are:

- Rarity
- Framework
- Results
- Search Behavior
- Dataset

On narrow screens, the links collapse behind a simple menu button. The menu updates `aria-expanded`, supports keyboard use, and remains useful without animation.

### 2. Publication Header

Use the centered Nerfies publication-header structure:

1. Full title: `Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking`
2. Parinthapat Pengpun, Simran Khanuja, and Graham Neubig, with existing personal links
3. NeuLab, Carnegie Mellon University
4. `EMNLP 2026`
5. Rounded Code and Dataset buttons

The header uses a conventional academic hierarchy. It does not use a separate oversized marketing title, eyebrow tags, gradient text, badges, or a metric strip.

### 3. Teaser

Place `task_example_figure.webp` directly below the publication header at a readable maximum width. Follow it with one short caption explaining that the model combines text, image evidence, and successive Wikipedia searches to identify the referenced entity.

### 4. Abstract

Reproduce the final manuscript abstract verbatim under the standard heading `Abstract`.

### 5. Rarity Beyond Popularity

Preserve the current website's complete rarity story, but derive the prose from the manuscript:

- Prior work commonly defines rare entities through popularity signals such as pageviews.
- The paper broadens this view with 15 popularity, documentation, knowledge-graph, and cross-lingual coverage metrics.
- The bottom-5% entity sets overlap by only 37% on average.
- Cultural Pangea accuracy drops by 15.4-39.9% across the rare-entity slices.
- Similar degradation can occur for entity sets that popularity metrics do not identify.
- Accuracy declines smoothly toward the sparse end, and the result is robust at 1%, 5%, and 10% thresholds.

Retain these figures:

- `metric_jaccard_heatmap.webp`
- `performance_drop_plot.webp`
- `decile_curves.webp`

The 15, 37%, 15.4-39.9%, and related values appear in the relevant prose or captions, not in a separate stat-card row.

### 6. Framework

Introduce the framework as simple and training-free. State that the model receives source-language text, an image, and a marked mention; reasons over the evidence; searches English Wikipedia; and returns one canonical English Wikipedia title.

Preserve the implementation details currently explained on the website:

- The first search is forced because small models can skip optional tool use despite instructions.
- Later tool decisions are automatic.
- Retrieval is limited to 20 iterations.
- A second pass extracts the final title from the completed reasoning trace.

#### New Iterative-Retrieval Diagram

Create a new responsive SVG or equivalent web-native figure with two coordinated regions.

The left region shows the reusable search loop:

1. Text + image + mention enter the reasoning VLM.
2. The reasoning VLM produces tool call `q_t`.
3. Wikipedia retrieval returns evidence `e_t`.
4. Evidence returns to the reasoning VLM and updates its state.
5. The next query `q_{t+1}` can differ from `q_t`.
6. When the evidence is sufficient, the system performs title extraction.

The right region unfolds the task example as a tool trace:

- Iteration 1 calls `search("cruise ship Yokohama virus")` and returns generic evidence that does not resolve the target.
- The model combines that evidence with the image, whose hull reads `Diamond Princess`.
- A dashed horizontal rule marks the next iteration.
- Iteration 2 calls `search("Diamond Princess ship")`, visibly different from the first query.
- Retrieval returns `Diamond Princess (ship)` with its description.
- A final dashed rule precedes the canonical output title.

The diagram's main claim is that retrieval results change the next query and therefore change the content that can be retrieved. It must not look like a generic three-box RAG pipeline. Styling should resemble a polished paper figure, using restrained versions of the paper's blue, peach, and green colors.

### 7. Reasoning and Retrieval

Preserve the controlled interaction analysis:

- Reasoning alone does not significantly improve rare-entity accuracy.
- Retrieval improves rare-entity accuracy even without reasoning but can hurt overall accuracy.
- Their combination performs best.
- The matched Qwen variants establish the controlled interaction.
- GLM confirms a rare-entity retrieval benefit in thinking mode but cannot test the complete interaction because its non-thinking mode could not sustain the loop.

Present the existing 2-by-2 accuracy comparison as a simple academic table or compact figure, not as colored dashboard cards.

### 8. Results

Preserve the current results content and figures:

- 87.9% average accuracy
- +6.9% over Cultural Pangea overall
- Gains up to +23.3% on rare-entity slices
- 14 of 15 rare-slice gains larger than the full-set gain
- A 4B reasoning model with embedding retrieval matches the 8B instruct model overall and improves on several rare slices
- Threshold-robust comparisons already represented on the page

Retain these figures:

- `intro_figure.webp`
- `rare_fig_r1_advantage_growth.webp`
- `rare_fig_r2_robustness_scatter.webp`
- `rare_fig_r6_reasoning_vs_size.webp`

### 9. Search Behavior

Preserve both current search-behavior findings:

- Embedding retrieval has substantially higher early-search hit rates than BM25, with the strongest refinement typically occurring on the second search.
- Instruct-model query diversity declines and verbatim repetition rises over long search chains, while strong reasoning models generally stop earlier.

Retain:

- `retrieval_fig_n4_retrieval_success.webp`
- `retrieval_fig_n6b_query_evolution.webp`

### 10. MERLIN-Rare

Retain the dataset description, 1,105 entity mentions, 790 unique images, five languages, 105 prediction sets, Hugging Face loading example, Dataset link, and repository guide link.

Use a normal paper-page section with a compact statistics line and code block. Do not use metric cards or a simulated terminal window.

### 11. Citation

Retain the BibTeX entry and a small Copy BibTeX control. The control reports its copied state accessibly and does not require a framework.

### 12. Acknowledgments and Footer

Retain exactly:

> We thank Ibrahim AlRayes for his help with this project, and Jean de Dieu Nyandwi and Zaid Sheikh for sharing resources that supported this work. We also thank the members of NeuLab for their helpful feedback.

> This work was supported in part by a research grant from the Defence Science and Technology Agency (DSTA), Singapore.

The footer includes Code, Dataset, website licensing information, and visible attribution linking to the Nerfies website template as requested by its CC BY-SA 4.0 terms.

## Visual System

- White page background and conventional academic spacing
- Google Sans or a close system fallback for publication headings
- Noto Sans or a close system fallback for body text
- Standard blue links
- Dark rounded publication-resource buttons
- Narrow reading measure for prose and wider measure for figures
- Minimal separators and no ornamental gradients
- Existing figure colors remain unchanged
- The new framework diagram uses restrained paper colors rather than interface styling

Remove the current visual language:

- floating translucent navigation
- gradient hero text and glows
- badges and pills used as decoration
- metric cards
- dark full-width result sections
- reveal-on-scroll animation
- lightbox treatment
- active-section scroll tracking
- marketing-style artifact cards

## Figure Layout

Choose the strongest layout during implementation rather than forcing one pattern everywhere.

- Use full width for the task example, new framework diagram, overlap heatmap, and main rare-entity result when readability benefits.
- Pair related figures only when both remain legible, such as baseline degradation with decile robustness and retrieval success with query evolution.
- Stack every paired figure on mobile.
- Give every figure an accurate alt attribute and a short paper-like caption stating what is measured and the relevant result.
- Do not duplicate the same number in a heading, caption, and body paragraph without need.

## Technical Architecture

The site remains a static GitHub Pages deployment.

- Base the HTML structure on the Nerfies template and Bulma.
- Vendor only the template assets that are actually used.
- Do not import Nerfies videos, carousel code, interpolation code, Google Analytics, jQuery, or research-specific assets.
- Use a small dependency-free script for the mobile navigation and Copy BibTeX behavior.
- Preserve `.nojekyll`, existing favicons, canonical URL, Open Graph metadata, Twitter metadata, and scholarly JSON-LD.
- Preserve local image dimensions in HTML so layout remains stable and the existing image-dimension test remains meaningful.

The authoritative site state is the committed static files under `docs/`. There is no runtime API or mutable server state.

## Accessibility and Responsive Behavior

- Keep one `h1` and a logical heading hierarchy.
- Preserve the skip link and semantic `header`, `nav`, `main`, `section`, and `footer` landmarks.
- Preserve descriptive alt text for every local image.
- Keep visible keyboard focus.
- Make the mobile navigation keyboard accessible and synchronize `aria-expanded` with its open state.
- Ensure buttons and links have usable touch targets.
- Respect reduced-motion preferences, although the redesigned page has no decorative motion.
- At narrow widths, stack columns, figures, author blocks, and resource buttons without clipping or horizontal scrolling.
- Essential information must remain available without JavaScript. Only the collapsed mobile menu and Copy BibTeX enhancement depend on it.

## Failure Semantics

- Broken local references fail the static reference test.
- Missing or incorrectly declared image dimensions fail the image-dimension test.
- If JavaScript is unavailable, all paper content, figures, external links, BibTeX, and acknowledgments remain visible.
- External Code and Dataset links use stable canonical URLs.
- The Paper button remains absent until a verified arXiv URL is available.

## Verification

Before publication:

1. Run the complete repository test suite.
2. Run `git diff --check`.
3. Validate every local reference and anchor.
4. Confirm every figure's declared dimensions match the asset.
5. Check external Code, Dataset, author, NeuLab, license, and template-attribution links.
6. Render and inspect the complete page at desktop width.
7. Render and inspect the complete page at mobile width.
8. Verify no overlap, clipping, unreadable figure text, broken navigation, or horizontal scrolling.
9. Test the mobile menu and Copy BibTeX behavior.
10. Inspect the final diff and verify the deployed GitHub Pages build.

## Intentional Deferred Work

- Add Paper and arXiv resource buttons after the arXiv page is live.
- Refine the new iterative-retrieval diagram during implementation while preserving the approved left-loop/right-trace information structure.
