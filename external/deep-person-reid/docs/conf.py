
import os
import sys

sys.path.insert(0, os.path.abspath(".."))



project = "torchreid"
copyright = "2019, Kaiyang Zhou"
author = "Kaiyang Zhou"

version_file = "../torchreid/__init__.py"
with open(version_file, "r") as f:
    exec(compile(f.read(), version_file, "exec"))
__version__ = locals()["__version__"]


version = __version__

release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_markdown_tables",
]

templates_path = ["_templates"]

source_suffix = [".rst", ".md"]

source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}

master_doc = "index"

language = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = None

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

htmlhelp_basename = "torchreiddoc"


latex_elements = {

}

latex_documents = [
    (master_doc, "torchreid.tex", "torchreid Documentation", "Kaiyang Zhou", "manual"),
]

man_pages = [(master_doc, "torchreid", "torchreid Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "torchreid",
        "torchreid Documentation",
        author,
        "torchreid",
        "One line description of project.",
        "Miscellaneous",
    ),
]


epub_title = project

epub_exclude_files = ["search.html"]

