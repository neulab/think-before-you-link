from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

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
        for name in ("href", "src"):
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
        with Image.open(DOCS / source) as image:
            assert image.size == (
                int(attributes["width"]),
                int(attributes["height"]),
            )


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
    }
    published = {
        image["src"]
        for image in parse_site().images
        if image.get("src", "").startswith("assets/figures/")
    }
    assert published == expected


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
    forbidden = [
        "hero-glow",
        "stat-grid",
        "section-dark",
        "data-lightbox",
        "reveal",
    ]
    assert all(token not in html for token in forbidden)


def test_navigation_and_diagram_have_accessible_contracts():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'class="skip-link"' in html
    assert 'aria-controls="site-menu"' in html
    assert 'aria-expanded="false"' in html
    assert 'id="site-menu"' in html
    assert 'role="img"' in html
    assert '<title id="framework-diagram-title">' in html
    assert '<desc id="framework-diagram-description">' in html
