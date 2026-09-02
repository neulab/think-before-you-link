from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

from PIL import Image


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"


class SiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.images = []
        self.classes = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if "class" in attributes:
            self.classes.extend(attributes["class"].split())
        for name in ("href", "src", "srcset"):
            if name in attributes:
                self.references.append(attributes[name])
        if tag == "img":
            self.images.append(attributes)

    def handle_data(self, data):
        self.text.append(data)


def parse_site():
    parser = SiteParser()
    parser.feed((DOCS / "index.html").read_text(encoding="utf-8"))
    return parser


def test_site_references_and_anchors_resolve():
    parser = parse_site()
    assert len(parser.ids) == len(set(parser.ids))
    ids = set(parser.ids)

    for reference in parser.references:
        if reference.startswith("#"):
            assert reference[1:] in ids
        elif not urlparse(reference).scheme:
            assert (DOCS / reference).is_file()


def test_published_image_dimensions_are_exact():
    for attributes in parse_site().images:
        source = attributes.get("src")
        if not source or urlparse(source).scheme:
            continue
        path = DOCS / source
        expected = (int(attributes["width"]), int(attributes["height"]))
        if path.suffix == ".svg":
            root = ElementTree.parse(path).getroot()
            viewbox = tuple(float(value) for value in root.attrib["viewBox"].split())
            assert viewbox[:2] == (0, 0)
            assert viewbox[2:] == expected
        else:
            with Image.open(path) as image:
                assert image.size == expected


def test_github_pages_bypasses_jekyll():
    assert (DOCS / ".nojekyll").is_file()


def test_site_uses_academic_publication_structure_and_preserves_content():
    parser = parse_site()
    text = " ".join(" ".join(parser.text).split())
    required = [
        "Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking",
        "EMNLP 2026",
        "Abstract",
        "Rarity Beyond Popularity",
        "Framework",
        "Reasoning and Retrieval",
        "Results",
        "Search Behavior",
        "MERLIN-Rare",
        "15.4–39.9%",
        "37%",
        "+23.3%",
        "Ibrahim AlRayes",
        "Defence Science and Technology Agency",
        "Nerfies",
    ]
    assert all(item in text for item in required)
    assert {
        "publication-title",
        "publication-authors",
        "publication-links",
    } <= set(parser.classes)


def test_site_preserves_every_published_figure():
    expected = {
        path.relative_to(DOCS).as_posix()
        for path in (DOCS / "assets" / "figures").glob("*.webp")
        if path.name != "task_example_figure.webp"
    }
    published = {
        image["src"]
        for image in parse_site().images
        if image.get("src", "").startswith("assets/figures/")
    }
    assert expected <= published


def test_results_use_horizontal_robustness_gap_chart():
    images = parse_site().images
    chart = next(
        image
        for image in images
        if image.get("src")
        == "assets/figures/rare_fig_r2_robustness_gap.svg"
    )

    alt = chart.get("alt", "").lower()
    assert "horizontal bar chart" in alt
    assert "14 of 15" in alt
    assert all(
        image.get("src")
        != "assets/figures/rare_fig_r2_robustness_scatter.webp"
        for image in images
    )


def test_site_has_approved_links_and_iterative_diagram():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "https://github.com/neulab/think-before-you-link" in html
    assert "https://huggingface.co/datasets/neulab/merlin-rare" in html
    assert "arxiv.org" not in html.lower()
    diagram_path = DOCS / "assets" / "figures" / "framework_diagram.svg"
    assert diagram_path.is_file()
    diagram = diagram_path.read_text(encoding="utf-8")
    assert "q_t" in diagram and "q_{t+1}" in diagram
    assert all(f"Iteration {number}" in diagram for number in (1, 2, 3))
    assert "stroke-dasharray" in diagram


def test_hero_integrates_task_and_search_while_abstract_stays_text_only():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    hero_start = html.index('<section class="hero teaser">')
    hero_end = html.index("</section>", hero_start)
    hero = html[hero_start:hero_end]
    assert hero.count("<figure") == 1
    assert "task_example_figure.webp" in hero
    assert 'class="hero-example"' in hero
    assert "framework_diagram.svg" not in hero

    abstract_start = html.index('<section class="section abstract-section"')
    abstract_end = html.index("</section>", abstract_start)
    abstract = html[abstract_start:abstract_end]
    assert "abstract-layout" not in abstract
    assert "task_example_figure.webp" not in abstract
    assert abstract.count("<figure") == 0


def test_framework_diagram_remains_unchanged_in_framework_section():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert html.count('src="assets/figures/framework_diagram.svg"') == 1
    assert html.count('srcset="assets/figures/framework_diagram_mobile.svg"') == 1

    for filename in ("framework_diagram.svg", "framework_diagram_mobile.svg"):
        diagram = (DOCS / "assets" / "figures" / filename).read_text(
            encoding="utf-8"
        )
        required = [
            "iteration 1",
            "iteration 2",
            "iteration 3",
            "reasoning",
            "search",
            "returned results",
            "cruise ship yokohama virus",
            "diamond princess",
            "yokohama",
            "diamond princess ship",
            "diamond princess (ship)",
        ]
        assert all(item in diagram.lower() for item in required)


def test_hero_diagram_shows_task_three_interleaved_rounds_and_linked_entity():
    required = [
        "input",
        "cruise ship",
        'image clue: “diamond princess”',
        "illustrative reasoning and search",
        "iteration 1",
        "iteration 2",
        "iteration 3",
        "reasoning",
        "search",
        "returned pages",
        "cruise ship yokohama virus",
        "diamond princess yokohama",
        "diamond princess ship",
        "linked entity",
        "diamond princess (ship)",
        "english wikipedia title",
    ]

    html = (DOCS / "index.html").read_text(encoding="utf-8")
    hero_start = html.index('<section class="hero teaser">')
    hero_end = html.index("</section>", hero_start)
    hero = html[hero_start:hero_end]
    parser = SiteParser()
    parser.feed(hero)
    text = " ".join(" ".join(parser.text).split()).lower()
    assert all(item in text for item in required)
    assert hero.count('class="hero-animation-stage') == 4
    assert all(f'data-hero-stage="{stage}"' in hero for stage in range(1, 5))


def test_hero_caption_connects_dynamic_retrieval_to_rare_entities():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    hero_start = html.index('<section class="hero teaser">')
    hero_end = html.index("</section>", hero_start)
    hero = " ".join(html[hero_start:hero_end].split())

    expected = (
        "Reasoning lets the model narrow, broaden, or redirect each search based "
        "on evidence returned in earlier rounds. Instead of retrieving a candidate "
        "set once and only reranking it, the model dynamically retrieves new "
        "evidence throughout the reasoning process. We show that this training-free "
        "approach recovers much of the performance lost on rare entities."
    )
    assert expected in hero


def test_hero_animation_plays_once_immediately_and_keeps_final_state():
    css = (DOCS / "static" / "css" / "index.css").read_text(encoding="utf-8")
    javascript = (DOCS / "static" / "js" / "index.js").read_text(
        encoding="utf-8"
    )

    assert "animation-iteration-count: 1" in css
    assert "animation-fill-mode: both" in css
    assert "animation-iteration-count: infinite" not in css
    assert "animation-play-state: paused" not in css
    assert "IntersectionObserver" not in javascript
    assert "data-hero-animation" not in javascript


def test_hero_stacks_before_its_horizontal_grid_can_overflow():
    css = (DOCS / "static" / "css" / "index.css").read_text(encoding="utf-8")
    responsive_start = css.index("@media screen and (max-width: 1180px)")
    responsive_end = css.index("@media screen and (max-width: 1023px)")
    hero_responsive_css = css[responsive_start:responsive_end]

    assert ".hero-example-grid" in hero_responsive_css
    assert "grid-template-columns: 1fr" in hero_responsive_css
    assert ".hero-flow-arrow" in hero_responsive_css
    assert "transform: rotate(90deg)" in hero_responsive_css


def test_hero_horizontal_grid_centers_the_dominant_trace_panel():
    css = (DOCS / "static" / "css" / "index.css").read_text(encoding="utf-8")
    assert (
        "grid-template-columns: minmax(0, 3fr) 30px minmax(0, 5fr) "
        "30px minmax(0, 3fr)" in css
    )


def test_site_removes_previous_landing_page_ui():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    forbidden_classes = {
        "hero-glow",
        "stat-grid",
        "section-dark",
        "reveal",
    }
    assert forbidden_classes.isdisjoint(parse_site().classes)
    assert "data-lightbox" not in html


def test_navigation_and_diagram_have_accessible_contracts():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'class="skip-link"' in html
    assert 'aria-controls="site-menu"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="site-menu"' in html
    assert html.count('alt="Iterative Wikipedia retrieval') == 1
    assert html.count('alt="A Japanese cruise ship mention') == 1

    diagram = (DOCS / "assets" / "figures" / "framework_diagram.svg").read_text(
        encoding="utf-8"
    )
    assert 'role="img"' in diagram
    assert '<title id="framework-diagram-title">' in diagram
    assert '<desc id="framework-diagram-description">' in diagram
