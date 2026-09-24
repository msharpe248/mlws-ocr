"""Command-line entry point.

    mlws-ocr run configs/default.toml scan.png   # run the pipeline
    mlws-ocr run configs/default.toml doc.pdf --pdf-page 0
    mlws-ocr stages                              # list registered algorithms
    mlws-ocr inspect                             # browse runs/ in the browser
    mlws-ocr-ui [IMAGE]                          # the interactive workbench in the browser
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlws-ocr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a pipeline config over an image")
    p_run.add_argument("config", help="TOML run config")
    p_run.add_argument("image", help="input page image")
    p_run.add_argument("--runs-dir", default="runs")
    p_run.add_argument("--doc-id", default=None)
    p_run.add_argument("--pdf-page", type=int, default=0,
                       help="page number for PDF inputs (0-based)")
    p_run.add_argument("--doc-type", default=None,
                       choices=["letter", "book", "legal", "form", "newspaper", "magazine", "block"],
                       help="optional layout hint (never required); 'block' says the image "
                            "is one block of text -- a paragraph or a table handed in alone")

    sub.add_parser("stages", help="list registered stage implementations")

    p_ins = sub.add_parser("inspect", help="serve the inspector UI over a runs directory")
    p_ins.add_argument("--runs-dir", default="runs")
    p_ins.add_argument("--port", type=int, default=8330)

    args = parser.parse_args(argv)

    import mlws_ocr.cleanup  # noqa: F401  (registers built-in stages)
    import mlws_ocr.layout   # noqa: F401
    import mlws_ocr.glyph.components  # noqa: F401
    import mlws_ocr.recognize.stage   # noqa: F401
    import mlws_ocr.decode            # noqa: F401
    import mlws_ocr.adapt             # noqa: F401
    from mlws_ocr.core import registry

    if args.command == "stages":
        for name in registry.available():
            print(name)
        return 0

    if args.command == "inspect":
        from mlws_ocr.inspector.server import serve
        serve(runs_dir=args.runs_dir, port=args.port)
        return 0

    if args.command == "run":
        from mlws_ocr.core.config import load_config
        from mlws_ocr.core.runner import run_pipeline
        run_dir = run_pipeline(load_config(args.config), args.image,
                               runs_dir=args.runs_dir, doc_id=args.doc_id,
                               pdf_page=args.pdf_page, doc_type=args.doc_type)
        print(f"run written to {run_dir}")
        return 0

    return 1


def ui_main(argv=None) -> int:
    """Entry point for the ``mlws-ocr-ui`` executable: the interactive workbench
    (open a page, see and correct every stage, re-run from any stage, save).
    The read-only run inspector is ``mlws-ocr inspect``."""
    parser = argparse.ArgumentParser(prog="mlws-ocr-ui",
                                     description="the mlws-ocr workbench: an interactive page reader in the browser")
    parser.add_argument("image", nargs="?", help="an image or PDF to open straight away")
    parser.add_argument("--config", default="configs/neural.toml", help="engine profile")
    parser.add_argument("--doc-type", default=None, help="letter, legal, receipt, newspaper, ...")
    parser.add_argument("--runs-dir", default="runs", help="where uploads and saved sessions go")
    parser.add_argument("--port", type=int, default=8330)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)
    from mlws_ocr.workbench.server import serve
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()
    serve(image=args.image, config=args.config, port=args.port, runs_dir=args.runs_dir,
          doc_type=args.doc_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
