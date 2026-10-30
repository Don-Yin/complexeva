"""Sphinx configuration: Furo theme, Google docstrings via napoleon, examples via sphinx-gallery."""

from importlib import metadata

project = "complexeva"
author = "Don Yin"
release = metadata.version("complexeva")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]
myst_enable_extensions = ["html_image"]  # README icons are <img> tags with widths
exclude_patterns = ["_build", "papers"]
autodoc_typehints = "description"

html_theme = "furo"
html_title = f"complexeva {release}"
html_favicon = "figures/logo/icon.png"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "ignore_pattern": r"readme_icons\.py|validate\.py",  # figure scripts for the README, not tutorials
    "download_all_examples": False,
}


def _readme_img_paths(app, path, docname, content):
    """README <img> tags point at docs/figures/ from the repo root; :relative-images: only fixes Markdown images."""
    if path.name == "README.md":
        content[0] = content[0].replace('src="docs/', 'src="')


def setup(app):
    app.connect("include-read", _readme_img_paths)
