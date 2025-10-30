from __future__ import annotations
import json, pathlib, sys
import typer
import os
from rich.console import Console
from rich import box
from rich.table import Table
from importlib.metadata import version as dist_version, PackageNotFoundError

from .utils.hw import get_system_report
from .utils.config import load_config, data_dir, results_dir as get_results_dir
from .datasets import write_mini_dataset, load_corpus_jsonl
from .index.bm25 import build_bm25, search_bm25
from .index.faiss_index import build_faiss, search_faiss

from datetime import datetime
from .eval import retrieval as eval_retrieval
from .retrievers.bm25_retriever import BM25Retriever
from .retrievers.faiss_retriever import FaissRetriever
from .retrievers.hybrid import HybridRetriever

from .generators.prompts import build_prompt
from .generators.local_hf import LocalHFGenerator
from .eval.qa import score_example
from .utils.text import split_sentences

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import faiss
    faiss.omp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(1)  
    torch.set_num_interop_threads(1)
except Exception:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = typer.Typer(add_completion=False)
console = Console()


def _pkg_version(name: str) -> str:
    try:
        return dist_version(name)
    except PackageNotFoundError:
        return "unknown"

def _ensure_dataset(name: str, ddir: pathlib.Path) -> pathlib.Path:
    if name == "mini":
        return write_mini_dataset(ddir)
    raise typer.BadParameter(f"Unknown dataset: {name}")

def _print_hits(hits, docstore_path: pathlib.Path):
    # load docstore into memory once
    docs = {}
    with docstore_path.open() as f:
        for line in f:
            d = json.loads(line)
            docs[d["id"]] = d
    table = Table(title="Top results", box=box.SIMPLE)
    table.add_column("Rank", justify="right")
    table.add_column("DocID")
    table.add_column("Score")
    table.add_column("Title")
    for i, (doc_id, score) in enumerate(hits, start=1):
        title = docs.get(doc_id, {}).get("title", "")
        table.add_row(str(i), doc_id, f"{score:.4f}", title)
    console.print(table)


@app.command("version")
def _show_version():
    """Show PocketRAG and key dependency versions."""
    rows = {
        "pocketrag": _pkg_version("pocketrag"),
        "python": sys.version.split()[0],
    }
    for lib in ["torch", "transformers", "sentence-transformers", "faiss-cpu"]:
        rows[lib] = _pkg_version(lib)

    table = Table(title="Versions", box=box.SIMPLE)
    table.add_column("Package")
    table.add_column("Version")
    for k, v in rows.items():
        table.add_row(k, v)
    console.print("\n")
    console.print(table)

@app.command()
def hw():
    """Print hardware & framework capabilities (for result reproducibility)."""
    info = get_system_report()
    console.print_json(data=info)

@app.command()
def init(
    data_dir_opt: str = typer.Option("./data", "--data-dir", help="Directory for datasets & indexes"),
    results_dir: str = typer.Option("./results", "--results-dir", help="Directory to write benchmark outputs"),
):
    """Create standard folders and a default config file."""
    dd = pathlib.Path(data_dir_opt); rd = pathlib.Path(results_dir)
    dd.mkdir(parents=True, exist_ok=True); rd.mkdir(parents=True, exist_ok=True)
    cfg = {
        "data_dir": str(dd.resolve()),
        "results_dir": str(rd.resolve()),
        "seed": 42,
        "device": "auto"
    }
    (pathlib.Path("pocketrag.yaml")).write_text(json.dumps(cfg, indent=2))
    console.print(f"[green]Initialized.[/green] data_dir={dd} results_dir={rd} config=pocketrag.yaml")


data_app = typer.Typer(help="Dataset utilities")
index_app = typer.Typer(help="Index builders")
search_app = typer.Typer(help="Ad-hoc search over an index")

app.add_typer(data_app, name="data")
app.add_typer(index_app, name="index")
app.add_typer(search_app, name="search")

@data_app.command("fetch")
def data_fetch(
    name: str = typer.Option("mini", help="Dataset name (mini for built-in)"),
):
    """Materialize a dataset into your data_dir."""
    cfg = load_config()
    ddir = data_dir(cfg)
    out = _ensure_dataset(name, ddir)
    console.print(f"[green]Dataset ready:[/green] {out}")

@index_app.command("build")
def index_build(
    method: str = typer.Option(..., "--method", help="bm25 | faiss"),
    dataset: str = typer.Option("mini", "--dataset", help="Dataset name (mini)"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--model", help="Embedding model (faiss only)"),
):
    """Build an index for a dataset."""
    cfg = load_config()
    ddir = data_dir(cfg) / dataset
    corpus = load_corpus_jsonl(ddir / "corpus.jsonl")
    out_dir = ddir / "index" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "bm25":
        build_bm25(corpus, out_dir)
        console.print(f"[green]BM25 index built at[/green] {out_dir}")
    elif method == "faiss":
        build_faiss(corpus, out_dir, model_name=model_name)
        console.print(f"[green]FAISS index built at[/green] {out_dir}")
    else:
        raise typer.BadParameter("method must be one of: bm25, faiss")

@search_app.command("run")
def search_run(
    method: str = typer.Option(..., "--method", help="bm25 | faiss"),
    dataset: str = typer.Option("mini", "--dataset", help="Dataset name"),
    query: str = typer.Option(..., "--query", help="Query text"),
    k: int = typer.Option(5, "--k", help="Number of results"),
    model_name: str = typer.Option(None, "--model", help="Embedding model (faiss only; defaults to index meta)"),
):
    """Run a quick search against a built index and print top-k results."""
    cfg = load_config()
    base = data_dir(cfg) / dataset / "index" / method

    if method == "bm25":
        hits = search_bm25(base, query, k=k)
        docstore = base / "docstore.jsonl"
        _print_hits(hits, docstore)
    elif method == "faiss":
        hits = search_faiss(base, query, k=k, model_name=model_name)
        docstore = base / "docstore.jsonl"
        _print_hits(hits, docstore)
    else:
        raise typer.BadParameter("method must be one of: bm25, faiss")


eval_app = typer.Typer(help="Evaluation utilities")
app.add_typer(eval_app, name="eval")

def _load_queries_qrels(ddir: pathlib.Path):
    import json
    queries = []
    with (ddir / "queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            queries.append(q)
    with (ddir / "qrels.json").open() as f:
        qrels = json.load(f)
    return queries, qrels

@eval_app.command("retrieval")
def eval_retrieval_cmd(
    method: str = typer.Option(..., "--method", help="bm25 | faiss | hybrid"),
    dataset: str = typer.Option("mini", "--dataset", help="Dataset name"),
    k: int = typer.Option(5, "--k", help="Top-k for metrics"),
    model_name: str = typer.Option(None, "--model", help="FAISS/hybrid embedding model (defaults to index meta if omitted)"),
):
    """
    Evaluate retrieval metrics on all queries for a dataset and write results.
    """
    cfg = load_config()
    ddir = data_dir(cfg) / dataset
    idx_root = ddir / "index"
    if method == "bm25":
        ret = BM25Retriever(idx_root / "bm25")
    elif method == "faiss":
        ret = FaissRetriever(idx_root / "faiss", model_name=model_name)
    elif method == "hybrid":
        bm = BM25Retriever(idx_root / "bm25")
        fa = FaissRetriever(idx_root / "faiss", model_name=model_name)
        ret = HybridRetriever(bm, fa)
    else:
        raise typer.BadParameter("method must be one of: bm25, faiss, hybrid")

    queries, qrels_map = _load_queries_qrels(ddir)

    results_map = {}
    for q in queries:
        qid = q["id"]
        hits = ret.search(q["question"], k=max(100, k)) 
        results_map[qid] = hits

    summary = eval_retrieval.aggregate(qrels_map, results_map, ks=(1,3,5,10))
    hw = get_system_report()
    meta = {
        "dataset": dataset,
        "method": method,
        "k": k,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": model_name,
        "hardware": hw,
    }

    run_dir = (get_results_dir(cfg) / dataset / "retrieval" / method)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "per_query.jsonl").open("w") as f:
        import json
        for q in queries:
            qid = q["id"]
            f.write(json.dumps({
                "id": qid,
                "question": q["question"],
                "hits": results_map[qid]
            }) + "\n")

    out = {"summary": summary, "meta": meta}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    from rich.table import Table
    from rich import box
    table = Table(title=f"Retrieval metrics ({method}, dataset={dataset})", box=box.SIMPLE)
    table.add_column("Metric"); table.add_column("Score")
    for kname, val in summary.items():
        table.add_row(kname, f"{val:.4f}")
    console.print(table)
    console.print(f"[green]Wrote per-query to[/green] {run_dir/'per_query.jsonl'}")
    console.print(f"[green]Wrote summary to[/green] {run_dir/'metrics.json'}")


def _load_docstore(docstore_path: pathlib.Path) -> dict:
    ds = {}
    with docstore_path.open() as f:
        import json
        for line in f:
            d = json.loads(line)
            ds[d["id"]] = d
    return ds

@eval_app.command("qa")
def eval_qa_cmd(
    retriever: str = typer.Option("hybrid", "--retriever", help="bm25 | faiss | hybrid"),
    dataset: str = typer.Option("mini", "--dataset", help="Dataset name"),
    k: int = typer.Option(5, "--k", help="Top-k docs passed to generator"),
    gen_model: str = typer.Option("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--gen-model", help="HF model name"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|mps"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
):
    """
    End-to-end QA evaluation: retrieve -> generate -> score (EM/F1 + groundedness).
    """
    cfg = load_config()
    ddir = data_dir(cfg) / dataset
    idx_root = ddir / "index"

    if retriever == "bm25":
        ret = BM25Retriever(idx_root / "bm25")
        docstore = _load_docstore(idx_root / "bm25" / "docstore.jsonl")
        method_tag = "bm25"
    elif retriever == "faiss":
        ret = FaissRetriever(idx_root / "faiss")
        docstore = _load_docstore(idx_root / "faiss" / "docstore.jsonl")
        method_tag = "faiss"
    elif retriever == "hybrid":
        bm = BM25Retriever(idx_root / "bm25")
        fa = FaissRetriever(idx_root / "faiss")
        ret = HybridRetriever(bm, fa)

        docstore = _load_docstore(idx_root / "bm25" / "docstore.jsonl")
        method_tag = "hybrid"
    else:
        raise typer.BadParameter("retriever must be one of: bm25, faiss, hybrid")

    queries, _qrels = _load_queries_qrels(ddir)

    gen = LocalHFGenerator(model_name=gen_model, device_pref=device)

    results = []
    import json, re
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", gen_model)
    run_dir = (get_results_dir(cfg) / dataset / "qa" / f"{method_tag}__{safe_model}")
    run_dir.mkdir(parents=True, exist_ok=True)

    for q in queries:
        qid, question = q["id"], q["question"]
        refs = q.get("answers", [])
        hits = ret.search(question, k=max(50, k))

        ctx_pieces = []
        for doc_id, _score in hits[:k]:
            d = docstore.get(doc_id)
            if not d: continue
            text = d.get("text","")

            first_sents = split_sentences(text)[:3]
            ctx_pieces.append(f"[{doc_id}] {d.get('title','')}: " + " ".join(first_sents))
        context = "\n".join(ctx_pieces)
        prompt = build_prompt(question, context)
        pred = gen.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


        metrics = score_example(pred, refs, [docstore.get(d,{}).get("text","") for d,_ in hits[:k]])
        results.append({
            "id": qid,
            "question": question,
            "answers": refs,
            "retrieved": [doc_id for doc_id,_ in hits[:k]],
            "prediction": pred,
            "metrics": metrics
        })

    agg = {"EM": 0.0, "F1": 0.0, "grounded_precision": 0.0}
    if results:
        n = len(results)
        agg = {
            "EM": sum(r["metrics"]["EM"] for r in results)/n,
            "F1": sum(r["metrics"]["F1"] for r in results)/n,
            "grounded_precision": sum(r["metrics"]["grounded_precision"] for r in results)/n,
        }

    with (run_dir / "per_query.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    meta = {
        "dataset": dataset,
        "retriever": method_tag,
        "k": k,
        "model_name": gen_model,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "hardware": get_system_report(),
    }
    (run_dir / "metrics.json").write_text(json.dumps({"summary": agg, "meta": meta}, indent=2))

    from rich.table import Table
    from rich import box
    table = Table(title=f"QA metrics ({method_tag}, model={gen_model}, dataset={dataset})", box=box.SIMPLE)
    table.add_column("Metric"); table.add_column("Score")
    for kname, val in agg.items():
        table.add_row(kname, f"{val:.4f}")
    console.print(table)
    console.print(f"[green]Wrote per-query to[/green] {run_dir/'per_query.jsonl'}")
    console.print(f"[green]Wrote summary to[/green] {run_dir/'metrics.json'}")