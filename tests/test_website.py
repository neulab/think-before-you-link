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

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])
        for name in ("href", "src"):
            if name in attributes:
                self.references.append(attributes[name])
        if tag == "img":
            self.images.append(attributes)


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
