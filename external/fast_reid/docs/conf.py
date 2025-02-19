
import os
import sys
from unittest import mock
from sphinx.domains import Domain
from typing import Dict, List, Tuple


import sphinx_rtd_theme


class GithubURLDomain(Domain):


    name = "githuburl"
    ROOT = "https://github.com/JDAI-CV/fast-reid/tree/master"
    LINKED_DOC = ["tutorials/install", "tutorials/getting_started"]

    def resolve_any_xref(self, env, fromdocname, builder, target, node, contnode):
        github_url = None
        if not target.endswith("html") and target.startswith("../../"):
            url = target.replace("../", "")
            github_url = url
        if fromdocname in self.LINKED_DOC:

            github_url = target

        if github_url is not None:
            if github_url.endswith("MODEL_ZOO") or github_url.endswith("README"):
                
                github_url += ".md"
            print("Ref {} resolved to github:{}".format(target, github_url))
            contnode["refuri"] = self.ROOT + github_url
            return [("githuburl:any", contnode)]
        else:
            return []

from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath("../"))
os.environ["DOC_BUILDING"] = "True"
DEPLOY = os.environ.get("READTHEDOCS") == "True"



try:
    import torch  
except ImportError:
    for m in [
        "torch", "torchvision", "torch.nn", "torch.nn.parallel", "torch.distributed", "torch.multiprocessing", "torch.autograd",
        "torch.autograd.function", "torch.nn.modules", "torch.nn.modules.utils", "torch.utils", "torch.utils.data", "torch.onnx",
        "torchvision", "torchvision.ops",
    ]:
        sys.modules[m] = mock.Mock(name=m)
    sys.modules['torch'].__version__ = "1.5"

for m in [
    "cv2", "scipy", "portalocker", 
    "google", "google.protobuf", "google.protobuf.internal", "onnx",
]:
    sys.modules[m] = mock.Mock(name=m)

sys.modules["cv2"].__version__ = "3.4"

import fastreid 


project = "fastreid"
copyright = "2019-2020, fastreid contributors"
author = "fastreid contributors"


version = fastreid.__version__

release = version



needs_sphinx = "3.0"


extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]


napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

if DEPLOY:
    intersphinx_timeout = 10
else:

    intersphinx_timeout = 0.1
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

templates_path = ["_templates"]

source_suffix = [".rst", ".md"]


master_doc = "index"

language = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build", "README.md", "tutorials/README.md"]


pygments_style = "sphinx"


html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

htmlhelp_basename = "fastreiddoc"


latex_elements = {

}

latex_documents = [
    (master_doc, "fastreid.tex", "fastreid Documentation", "fastreid contributors", "manual")
]

man_pages = [(master_doc, "fastreid", "fastreid Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "fastreid",
        "fastreid Documentation",
        author,
        "fastreid",
        "One line description of project.",
        "Miscellaneous",
    )
]

todo_include_todos = True


def autodoc_skip_member(app, what, name, obj, skip, options):

    if getattr(obj, "__HIDE_SPHINX_DOC__", False):
        return True


    HIDDEN = {
        "GroupedBatchSampler",
    }
    try:
        if name in HIDDEN or (
            hasattr(obj, "__doc__") and obj.__doc__.lower().strip().startswith("deprecated")
        ):
            print("Skipping deprecated object: {}".format(name))
            return True
    except:
        pass
    return skip


_PAPER_DATA = {
    "resnet": ("1512.03385", "Deep Residual Learning for Image Recognition"),
    "fpn": ("1612.03144", "Feature Pyramid Networks for Object Detection"),
    "mask r-cnn": ("1703.06870", "Mask R-CNN"),
    "faster r-cnn": (
        "1506.01497",
        "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
    ),
    "deformconv": ("1703.06211", "Deformable Convolutional Networks"),
    "deformconv2": ("1811.11168", "Deformable ConvNets v2: More Deformable, Better Results"),
    "panopticfpn": ("1901.02446", "Panoptic Feature Pyramid Networks"),
    "retinanet": ("1708.02002", "Focal Loss for Dense Object Detection"),
    "cascade r-cnn": ("1712.00726", "Cascade R-CNN: Delving into High Quality Object Detection"),
    "lvis": ("1908.03195", "LVIS: A Dataset for Large Vocabulary Instance Segmentation"),
    "rrpn": ("1703.01086", "Arbitrary-Oriented Scene Text Detection via Rotation Proposals"),
    "imagenet in 1h": ("1706.02677", "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"),
    "xception": ("1610.02357", "Xception: Deep Learning with Depthwise Separable Convolutions"),
    "mobilenet": (
        "1704.04861",
        "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
    ),
}


def paper_ref_role(
    typ: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner,
    options: Dict = {},
    content: List[str] = [],
):
    """
    Parse :paper:`xxx`. Similar to the "extlinks" sphinx extension.
    """
    from docutils import nodes, utils
    from sphinx.util.nodes import split_explicit_title

    text = utils.unescape(text)
    has_explicit_title, title, link = split_explicit_title(text)
    link = link.lower()
    if link not in _PAPER_DATA:
        inliner.reporter.warning("Cannot find paper " + link)
        paper_url, paper_title = "#", link
    else:
        paper_url, paper_title = _PAPER_DATA[link]
        if "/" not in paper_url:
            paper_url = "https://arxiv.org/abs/" + paper_url
    if not has_explicit_title:
        title = paper_title
    pnode = nodes.reference(title, title, internal=False, refuri=paper_url)
    return [pnode], []


def setup(app):
    from recommonmark.transform import AutoStructify

    app.add_domain(GithubURLDomain)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.add_role("paper", paper_ref_role)
    app.add_config_value(
        "recommonmark_config",
        {"enable_math": True, "enable_inline_math": True, "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
