# Nerfies-Style Project Website Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bespoke GitHub Pages site with a restrained Nerfies/Bulma academic project page while preserving all published content and adding a clear iterative-retrieval diagram.

**Architecture:** The site remains a static document served from `docs/`. `docs/index.html` owns all semantic content and the inline responsive framework SVG. Vendored Bulma provides the base layout, a small custom stylesheet provides paper-specific presentation, and dependency-free JavaScript implements only the mobile menu and Copy BibTeX.

**Tech Stack:** Static HTML5, vendored Bulma CSS, custom CSS, vanilla JavaScript, GitHub Pages, pytest static-integrity tests, and desktop/mobile browser rendering.

**Spec:** `docs/superpowers/specs/2026-09-01-nerfies-website-redesign.md`

## Global Constraints

- Change website files under `docs/` and website integrity tests only.
- Use the final manuscript as the authority for claims, qualifications, and numbers.
- Preserve every substantive topic and figure from the current website.
- Show `EMNLP 2026` as a main-conference venue.
- Show Code and Dataset; omit Paper and arXiv until a verified URL exists.
- Lead with the existing task example.
- Use short paper-like headings and remove promotional language and `Finding N` labels.
- Remove stat cards, gradients, dark feature bands, reveal animation, active scroll tracking, and lightboxes.
- Preserve metadata, favicons, acknowledgments, DSTA support, responsive behavior, and accessibility.
- Credit the Nerfies template visibly in the footer.
- Do not add analytics, jQuery, video, carousel, slider, or unnecessary dependencies.

## File Structure

- Modify `tests/test_website.py` for structural, content, legacy-removal, and accessibility assertions.
- Modify `docs/index.html` for the complete page and inline framework SVG.
- Create `docs/static/css/bulma.min.css` as the unmodified template dependency.
- Create `docs/static/css/index.css` for project-specific presentation.
- Create `docs/static/js/index.js` for mobile navigation and Copy BibTeX.
- Delete `docs/styles.css` and `docs/script.js` after references move.
- Preserve `docs/assets/figures/*.webp`, `.nojekyll`, and favicon files unchanged.

---

### Task 1: Lock the Redesigned Page Contract in Tests

**Files:**
- Modify: `tests/test_website.py`
- Test: `tests/test_website.py`

**Interfaces:**
- Consumes: `docs/index.html` through the existing `SiteParser`.
- Produces: regression checks all later tasks must satisfy.

- [ ] **Step 1: Extend `SiteParser` to capture classes and text**

Add `self.classes`, `self.text`, class collection in `handle_starttag`, and:

```python
def handle_data(self, data):
    self.text.append(data)
```

Keep the existing ID, reference, and image collection logic unchanged.

- [ ] **Step 2: Add the publication and content contract test**

```python
def test_site_uses_academic_publication_structure_and_preserves_content():
    parser = parse_site()
    text = " ".join(" ".join(parser.text).split())
    required = [
        "Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking",
        "EMNLP 2026", "Abstract", "Rarity Beyond Popularity", "Framework",
        "Reasoning and Retrieval", "Results", "Search Behavior", "MERLIN-Rare",
        "15.4-39.9%", "37%", "+23.3%", "Ibrahim AlRayes",
        "Defence Science and Technology Agency", "Nerfies",
    ]
    assert all(item in text for item in required)
    assert {"publication-title", "publication-authors", "publication-links"} <= set(parser.classes)
```

- [ ] **Step 3: Add link, diagram, and legacy-removal tests**

```python
def test_site_has_approved_links_and_iterative_diagram():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "https://github.com/neulab/think-before-you-link" in html
    assert "https://huggingface.co/datasets/neulab/merlin-rare" in html
    assert "arxiv.org" not in html.lower()
    assert "q_t" in html and "q_{t+1}" in html
    assert "Iteration 1" in html and "Iteration 2" in html
    assert "stroke-dasharray" in html


def test_site_removes_previous_landing_page_ui():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    forbidden = ["hero-glow", "stat-grid", "section-dark", "data-lightbox", "reveal"]
    assert all(token not in html for token in forbidden)
```

- [ ] **Step 4: Add the accessibility contract test**

```python
def test_navigation_and_diagram_have_accessible_contracts():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'class="skip-link"' in html
    assert 'aria-controls="site-menu"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="site-menu"' in html
    assert 'role="img"' in html
    assert '<title id="framework-diagram-title">' in html
    assert '<desc id="framework-diagram-description">' in html
```

- [ ] **Step 5: Run tests and confirm the new contract fails against the old site**

Run:

```bash
env PYTHONPATH=. uv run --with-requirements requirements-dev.txt pytest tests/test_website.py -q
```

Expected: existing reference, image, and `.nojekyll` tests pass; new redesign tests fail.

- [ ] **Step 6: Commit the regression contract**

```bash
git add tests/test_website.py
git commit -m "Test academic project website contract"
```

---

### Task 2: Establish the Nerfies/Bulma Publication Shell

**Files:**
- Create: `docs/static/css/bulma.min.css`
- Create: `docs/static/css/index.css`
- Create: `docs/static/js/index.js`
- Modify: `docs/index.html`
- Delete: `docs/styles.css`
- Delete: `docs/script.js`
- Test: `tests/test_website.py`

**Interfaces:**
- Consumes: metadata, author URLs, Code/Dataset URLs, favicons, and `task_example_figure.webp`.
- Produces: the semantic publication header, responsive navigation, teaser, and exact abstract.

- [ ] **Step 1: Vendor only Bulma from the audited Nerfies template**

Copy `/private/tmp/nerfies-template/static/css/bulma.min.css` verbatim to `docs/static/css/bulma.min.css`. Do not copy analytics, videos, carousel, slider, jQuery, Font Awesome, Academicons, or interpolation files.

Verify:

```bash
cmp /private/tmp/nerfies-template/static/css/bulma.min.css docs/static/css/bulma.min.css
```

Expected: exit status 0.

- [ ] **Step 2: Replace `docs/index.html` with the Nerfies-style shell**

Preserve description, Open Graph, Twitter, canonical, favicon, and scholarly JSON-LD metadata. Point to the new CSS/JS paths. The shell begins:

```html
<a class="skip-link" href="#main-content">Skip to content</a>
<nav class="navbar project-navbar" aria-label="Page sections">
  <button class="navbar-burger" type="button" aria-label="Open page navigation"
          aria-expanded="false" aria-controls="site-menu">
    <span aria-hidden="true"></span><span aria-hidden="true"></span><span aria-hidden="true"></span>
  </button>
  <div class="navbar-menu" id="site-menu">
    <a class="navbar-item" href="#rarity">Rarity</a>
    <a class="navbar-item" href="#framework">Framework</a>
    <a class="navbar-item" href="#results">Results</a>
    <a class="navbar-item" href="#search-behavior">Search Behavior</a>
    <a class="navbar-item" href="#dataset">Dataset</a>
  </div>
</nav>
```

Use the centered Nerfies publication header with the full title, linked authors, NeuLab/CMU, `EMNLP 2026`, and Code/Dataset rounded dark buttons. Do not add a Paper button.

- [ ] **Step 3: Add the teaser and exact manuscript abstract**

Use `assets/figures/task_example_figure.webp` with actual width/height attributes and descriptive alt text. Follow it with one sentence. Reproduce `latex/sections/0-abstract.tex` verbatim under `Abstract`.

- [ ] **Step 4: Implement minimal custom CSS**

In `docs/static/css/index.css`, use Google Sans for publication headings and Noto Sans for body text, standard blue author links, dark rounded resource buttons, a narrow prose measure, wide figure measure, visible focus, and one mobile breakpoint. Do not add gradients, shadows, translucent floating surfaces, reveals, or dark section themes.

- [ ] **Step 5: Implement dependency-free mobile navigation**

```javascript
const burger = document.querySelector(".navbar-burger");
const menu = document.querySelector("#site-menu");

burger?.addEventListener("click", () => {
  const expanded = burger.getAttribute("aria-expanded") === "true";
  burger.setAttribute("aria-expanded", String(!expanded));
  burger.classList.toggle("is-active", !expanded);
  menu?.classList.toggle("is-active", !expanded);
});
```

Close the menu and reset `aria-expanded` when a menu link is selected.

- [ ] **Step 6: Delete the old entry points and run tests**

Delete `docs/styles.css` and `docs/script.js` only after `index.html` references the new files. Run:

```bash
env PYTHONPATH=. uv run --with-requirements requirements-dev.txt pytest tests/test_website.py -q
```

Expected: structural tests pass; content tests may fail only for sections added in Tasks 3-4.

- [ ] **Step 7: Commit the publication shell**

```bash
git add docs/index.html docs/static docs/styles.css docs/script.js
git commit -m "Adopt Nerfies academic website shell"
```

---

### Task 3: Add Rarity, Framework, and Iterative Retrieval Diagram

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/static/css/index.css`
- Test: `tests/test_website.py`

**Interfaces:**
- Consumes: Task 2 shell, manuscript prose, and existing rarity figures.
- Produces: complete rarity and framework sections plus an accessible responsive SVG.

- [ ] **Step 1: Add `Rarity Beyond Popularity` from the manuscript**

Create `<section class="section" id="rarity">` using concise prose from `latex/sections/1-introduction.tex` and `latex/sections/4-rarity.tex`. Include 15 metrics, 37% overlap, 15.4-39.9% degradation, and 1%/5%/10% robustness in relevant paragraphs and captions.

- [ ] **Step 2: Add all three rarity figures**

Include:

```text
assets/figures/metric_jaccard_heatmap.webp
assets/figures/performance_drop_plot.webp
assets/figures/decile_curves.webp
```

Read actual dimensions with Pillow before writing `width` and `height`. Show the overlap heatmap full width. Pair degradation and decile robustness only if both remain readable; otherwise show each full width. Stack all paired figures on mobile.

- [ ] **Step 3: Add framework prose and operational details**

Create `<section class="section" id="framework">` from `latex/sections/4-methodology.tex`. Include training-free operation, forced first search, automatic later tool decisions, the 20-iteration limit, and second-pass title extraction.

- [ ] **Step 4: Implement the approved framework SVG**

Use:

```html
<svg class="framework-diagram" viewBox="0 0 1200 620" role="img"
     aria-labelledby="framework-diagram-title framework-diagram-description">
  <title id="framework-diagram-title">Iterative Wikipedia retrieval and example tool trace</title>
  <desc id="framework-diagram-description">
    Returned evidence updates the model's reasoning state, allowing each subsequent search query to retrieve different content.
  </desc>
</svg>
```

The left region must show input -> reasoning VLM <-> Wikipedia retrieval -> title extraction. Label the outgoing tool call `q_t`, the returned evidence `e_t`, and state that `q_{t+1}` can differ after evidence updates the reasoning state.

The right region must show:

```text
Iteration 1
search("cruise ship Yokohama virus")
Generic cruise-ship evidence; target unresolved
Image evidence: hull reads "Diamond Princess"

Iteration 2
search("Diamond Princess ship")
Diamond Princess (ship)
Grand-class cruise ship; COVID-19 outbreak

Final output
Diamond Princess (ship)
```

Separate iterations with dashed horizontal rules. Make the two tool calls visibly different. Use restrained paper-like blue, peach, green, and neutral fills. Do not collapse the concept into a generic three-box RAG pipeline.

- [ ] **Step 5: Make the diagram readable on mobile**

Use a desktop side-by-side arrangement. Below the mobile breakpoint, stack the loop and trace or provide a coordinated mobile SVG. Keep visible diagram labels at least 12 CSS pixels and prevent horizontal page scrolling.

- [ ] **Step 6: Run tests and commit**

```bash
env PYTHONPATH=. uv run --with-requirements requirements-dev.txt pytest tests/test_website.py -q
git add docs/index.html docs/static/css/index.css
git commit -m "Add rarity and iterative retrieval story"
```

Expected: rarity and diagram assertions pass. Any remaining failures identify only Task 4 content.

---

### Task 4: Add Results, Search Behavior, Dataset, and Citation

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/static/css/index.css`
- Modify: `docs/static/js/index.js`
- Test: `tests/test_website.py`

**Interfaces:**
- Consumes: Task 3 page, manuscript, result assets, Code URL, and Dataset URL.
- Produces: the complete content-preserving page and all final navigation targets.

- [ ] **Step 1: Add `Reasoning and Retrieval`**

Present the Qwen3-VL 8B comparison as a simple academic table:

```text
                 No retrieval    Embedding retrieval
Instruct              83.5              82.9
Thinking              84.2              87.9
```

Follow with manuscript-derived prose explaining the controlled interaction and the limited GLM evidence boundary. Do not use dashboard cards or branded callouts.

- [ ] **Step 2: Add `Results` and all four result figures**

Preserve 87.9%, +6.9%, +23.3%, the 14-of-15 result, and the 4B-versus-8B comparison. Include:

```text
assets/figures/intro_figure.webp
assets/figures/rare_fig_r1_advantage_growth.webp
assets/figures/rare_fig_r2_robustness_scatter.webp
assets/figures/rare_fig_r6_reasoning_vs_size.webp
```

Use the first two full width. Pair supporting figures only if their rendered labels remain legible.

- [ ] **Step 3: Add `Search Behavior`**

Explain early embedding-retrieval success, second-search refinement, diminishing later returns, instruct-model query degradation, and earlier stopping by strong reasoning models. Include and pair on desktop:

```text
assets/figures/retrieval_fig_n4_retrieval_success.webp
assets/figures/retrieval_fig_n6b_query_evolution.webp
```

Stack them on mobile.

- [ ] **Step 4: Add `MERLIN-Rare`**

Retain 1,105 entity mentions, 790 unique images, five languages, 105 prediction sets, the Hugging Face loading example, Dataset link, and repository guide. Use ordinary prose, one compact statistics sentence, and a plain code block.

- [ ] **Step 5: Add BibTeX and progressive enhancement**

Use `<pre id="bibtex-code"><code>...</code></pre>` and `<button id="copy-bibtex" type="button">Copy BibTeX</button>`. Append:

```javascript
const copyButton = document.querySelector("#copy-bibtex");
const bibtex = document.querySelector("#bibtex-code");

copyButton?.addEventListener("click", async () => {
  if (!bibtex) return;
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    copyButton.textContent = "Copied";
    window.setTimeout(() => { copyButton.textContent = "Copy BibTeX"; }, 1600);
  } catch {
    copyButton.textContent = "Select and copy the citation";
  }
});
```

The citation stays visible and selectable without JavaScript or clipboard permission.

- [ ] **Step 6: Add acknowledgments and footer attribution**

Include both approved acknowledgment paragraphs verbatim. Add Code, Dataset, CC BY-SA 4.0 website licensing, and `Website template adapted from Nerfies` linking to `https://github.com/nerfies/nerfies.github.io`.

- [ ] **Step 7: Run all tests and scan for legacy UI**

```bash
env PYTHONPATH=. uv run --with-requirements requirements-dev.txt pytest -q
rg -n 'Finding [0-9]|hero-glow|stat-grid|section-dark|data-lightbox|reveal' docs/index.html docs/static
```

Expected: all tests pass; the scan returns no matches.

- [ ] **Step 8: Commit complete content migration**

```bash
git add docs/index.html docs/static tests/test_website.py
git commit -m "Migrate complete paper content to academic layout"
```

---

### Task 5: Responsive, Visual, and Live Verification

**Files:**
- Modify if defects are found: `docs/index.html`
- Modify if defects are found: `docs/static/css/index.css`
- Modify if defects are found: `docs/static/js/index.js`
- Modify if a regression is found: `tests/test_website.py`

**Interfaces:**
- Consumes: complete Task 4 website.
- Produces: a visually verified, accessible, deployed GitHub Pages site.

- [ ] **Step 1: Start a local static server**

```bash
python3 -m http.server 6420 --bind 127.0.0.1 --directory docs
```

- [ ] **Step 2: Inspect a complete desktop rendering**

At approximately 1440 CSS pixels wide, inspect the full page. Verify centered title/authors/venue/buttons, readable task example, legible paired figures, clear framework diagram, conventional paper-page spacing, and absence of residual landing-page styling.

- [ ] **Step 3: Inspect a complete mobile rendering**

At approximately 390 CSS pixels wide, verify no horizontal page scroll, clean title/author wrapping, functional menu with correct `aria-expanded`, adequate touch targets, stacked figures, readable diagram labels, and internally scrolling code/BibTeX.

- [ ] **Step 4: Test interaction and no-JavaScript behavior**

With JavaScript, test the mobile menu and Copy BibTeX. Without JavaScript, confirm all content, figures, links, citation, and acknowledgments remain visible.

- [ ] **Step 5: Check external links**

Require successful responses from Code, Dataset, all three author pages, NeuLab, CC BY-SA 4.0, and the Nerfies template repository.

- [ ] **Step 6: Run final verification**

```bash
env PYTHONPATH=. uv run --with-requirements requirements-dev.txt pytest -q
git diff --check
git status --short
git diff --stat
```

Expected: all tests pass, no whitespace errors, and only planned website/test files differ.

- [ ] **Step 7: Commit visual fixes if needed**

```bash
git add docs/index.html docs/static tests/test_website.py
git commit -m "Polish responsive academic website"
```

Do not create an empty commit if inspection requires no changes.

- [ ] **Step 8: Reconcile, push, and verify GitHub Pages**

Fetch before pushing. Push `main`, verify remote `main` equals local `HEAD`, confirm the Pages build succeeds, and confirm the public page contains `EMNLP 2026`, `Iteration 2`, and the DSTA statement while containing no arXiv link or legacy landing-page class.

---

## Plan Self-Review Results

- **Spec coverage:** Tasks 1-5 cover the template shell, complete content, iterative diagram, removed UI, accessibility, mobile behavior, citation, acknowledgments, licensing, and live verification.
- **Placeholder scan:** No placeholder marker, unspecified error-handling instruction, or incomplete implementation step remains. The only deferred item is the explicitly approved future arXiv link.
- **Interface consistency:** Navigation IDs remain `rarity`, `framework`, `results`, `search-behavior`, and `dataset`. Script selectors remain `.navbar-burger`, `#site-menu`, `#copy-bibtex`, and `#bibtex-code`. Diagram IDs remain `framework-diagram-title` and `framework-diagram-description`.
