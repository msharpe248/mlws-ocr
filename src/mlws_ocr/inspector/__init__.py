"""The inspector: a local web viewer for persisted pipeline runs.

"Eyes on every part of the processing" is a project requirement, not a
nicety: the viewer renders every stage's DebugBundle straight from the
``runs/`` directory, so no pipeline code ever needs a UI dependency.
"""
