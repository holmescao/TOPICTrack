
import os
import sys
from unittest import mock
from sphinx.domains import Domain
from typing import Dict, List, Tuple


import sphinx_rtd_theme


class GithubURLDomain(Domain):
    

    name = "githuburl"
    ROOT = "https://github.com/Megvii-BaseDetection/YOLOX"
   
    LINKED_DOC = [
        "tutorials/install",
    ]

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
os.environ["_DOC_BUILDING"] = "True"
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
    sys.modules['torch'].__version__ = "1.7" 
    HAS_TORCH = False
else:
    try:
        torch.ops.yolox = mock.Mock(name="torch.ops.yolox")
    except:
        pass
    HAS_TORCH = True

for m in [
    "cv2", "scipy", "portalocker", "yolox._C",
    "pycocotools", "pycocotools.mask", "pycocotools.coco", "pycocotools.cocoeval",
    "google", "google.protobuf", "google.protobuf.internal", "onnx",
    "caffe2", "caffe2.proto", "caffe2.python", "caffe2.python.utils", "caffe2.python.onnx", "caffe2.python.onnx.backend",
]:
    sys.modules[m] = mock.Mock(name=m)

sys.modules["cv2"].__version__ = "3.4"

import yolox  



project = "YOLOX"
copyright = "2021-2021, YOLOX contributors"
author = "YOLOX contributors"

version = yolox.__version__

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
    "sphinx_markdown_tables",
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

    intersphinx_timeout = 0.5
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

templates_path = ["_templates"]

source_suffix = [".rst", ".md"]

master_doc = "index"

language = None

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "build",
    "README.md",
    "tutorials/README.md",
]

pygments_style = "sphinx"



html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

htmlhelp_basename = "yoloxdoc"



latex_elements = {
 
}

latex_documents = [
    (master_doc, "yolox.tex", "yolox Documentation", "yolox contributors", "manual")
]


man_pages = [(master_doc, "YOLOX", "YOLOX Documentation", [author], 1)]


texinfo_documents = [
    (
        master_doc,
        "YOLOX",
        "YOLOX Documentation",
        author,
        "YOLOX",
        "One line description of project.",
        "Miscellaneous",
    )
]

todo_include_todos = True


def autodoc_skip_member(app, what, name, obj, skip, options):

    if getattr(obj, "__HIDE_SPHINX_DOC__", False):
        return True

    HIDDEN = {
        "ResNetBlockBase",
        "GroupedBatchSampler",
        "build_transform_gen",
        "export_caffe2_model",
        "export_onnx_model",
        "apply_transform_gens",
        "TransformGen",
        "apply_augmentations",
        "StandardAugInput",
        "build_batch_data_loader",
        "draw_panoptic_seg_predictions",
        "WarmupCosineLR",
        "WarmupMultiStepLR",
    }
    try:
        if name in HIDDEN or (
            hasattr(obj, "__doc__")
            and obj.__doc__.lower().strip().startswith("deprecated")
        ):
            print("Skipping deprecated object: {}".format(name))
            return True
    except:
        pass
    return skip



def setup(app):
    from recommonmark.transform import AutoStructify

    app.add_domain(GithubURLDomain)
    app.connect("autodoc-skip-member", autodoc_skip_member)
   
    app.add_config_value(
        "recommonmark_config",
        {"enable_math": True, "enable_inline_math": True, "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
