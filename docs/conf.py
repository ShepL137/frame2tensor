project = "frame2tensor"
author  = "Shep L"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autodoc_member_order = "bysource"
autoclass_content    = "both"
autodoc_typehints    = "description"

intersphinx_mapping = {
    "python"  : ("https://docs.python.org/3", None),
    "torch"   : ("https://docs.pytorch.org/docs/stable/", None),
    "moderngl": ("https://moderngl.readthedocs.io/en/latest/", None),
}

exclude_patterns = ["_build"]

# moderngl's own Sphinx inventory registers its classes unprefixed (`Context`, not `moderngl.Context`).
# Even though `Context.__module__` really is `moderngl`, intersphinx cannot resolve our type hints against it.
nitpick_ignore_regex = [
    ("py:class", r"moderngl\..*"),
]

html_theme = "furo"
html_title = "frame2tensor"
