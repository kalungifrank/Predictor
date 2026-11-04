"""
Target Overlap & PPI Network
-----------------------------
Page 2 of the Inhibitor Predictor app.

Workflow:
    1. Provide a drug-target gene list and a disease gene list
    2. Draw a Venn diagram of the overlapping genes (with an enrichment test)
    3. Build a protein-protein interaction (PPI) network (STRING API or your own table)
    4. Rank the top interacting targets by network centrality

Run as part of the multi-page app:   streamlit run app.py
Or on its own:                       streamlit run pages/1_Target_Network.py
"""

import io
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import networkx as nx
import requests
from scipy.stats import hypergeom

# ======================================================================
# Sample data (illustrative gene lists, NOT curated from a database)
# ======================================================================

SAMPLE_DRUG_GENES = """PPARG
PRKAA1
PRKAA2
DPP4
SLC2A4
AKT1
MTOR
INSR
GCK
TNF
IL6
NFKB1
ADIPOQ
PTGS2
MAPK1
MAPK3
STAT3
SIRT1
PPARA
PPARGC1A"""

SAMPLE_DISEASE_GENES = """INS
INSR
IRS1
IRS2
PPARG
TCF7L2
KCNJ11
GCK
SLC2A4
AKT2
ADIPOQ
LEP
LEPR
TNF
IL6
FTO
HNF1A
HNF4A
GCG
GLP1R
CRP
SOD2"""

STRING_API = "https://string-db.org/api/tsv/network"

ORIGIN_COLORS = {
    "Drug ∩ Disease": "#D62728",
    "Drug only": "#4C78A8",
    "Disease only": "#F58518",
    "Added interactor": "#9E9E9E",
}

METRIC_LABELS = {
    "Degree": "degree",
    "Weighted degree": "weighted_degree",
    "Betweenness": "betweenness",
    "Closeness": "closeness",
    "Eigenvector": "eigenvector",
    "PageRank": "pagerank",
}

# ======================================================================
# Pure helper functions (no Streamlit UI calls)
# ======================================================================


def clean_gene_list(items):
    """Upper-case, strip, de-duplicate while preserving order."""
    seen, out = set(), []
    for g in items:
        g = str(g).strip().upper()
        if not g or g in {"NAN", "NONE", "NA"}:
            continue
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def parse_text_genes(text):
    return clean_gene_list(re.split(r"[\s,;]+", text or ""))


def overlap_stats(a, b, background=20000):
    """Overlap size, Jaccard index and hypergeometric enrichment p-value."""
    A, B = set(a), set(b)
    inter, union = A & B, A | B
    N = max(int(background), len(union), 1)
    k = len(inter)
    p = float(hypergeom.sf(k - 1, N, len(A), len(B))) if k > 0 else 1.0
    return {
        "overlap": k,
        "union": len(union),
        "jaccard": k / len(union) if union else 0.0,
        "expected": len(A) * len(B) / N,
        "p_value": p,
        "background": N,
    }


def draw_venn(a, b, a_label, b_label):
    """Two-set Venn diagram drawn with matplotlib (no extra dependency)."""
    A, B = set(a), set(b)
    c1, c2 = "#4C78A8", "#F58518"
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.add_patch(Circle((-0.55, 0), 1.0, facecolor=to_rgba(c1, 0.45), edgecolor=c1, lw=2))
    ax.add_patch(Circle((0.55, 0), 1.0, facecolor=to_rgba(c2, 0.45), edgecolor=c2, lw=2))
    kw = dict(ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(-1.15, 0, str(len(A - B)), **kw)
    ax.text(0.0, 0, str(len(A & B)), **kw)
    ax.text(1.15, 0, str(len(B - A)), **kw)
    lk = dict(ha="center", va="bottom", fontsize=12)
    ax.text(-0.75, 1.12, f"{a_label}\n(n = {len(A)})", **lk)
    ax.text(0.75, 1.12, f"{b_label}\n(n = {len(B)})", **lk)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.2, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    return buf.getvalue()


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_string_network(genes, species=9606, required_score=400, add_nodes=0):
    """Query the STRING API and return an edge table: source, target, weight (0-1)."""
    params = {
        "identifiers": "\r".join(genes),
        "species": int(species),
        "required_score": int(required_score),
        "caller_identity": "inhibitor_predictor_app",
    }
    if add_nodes and add_nodes > 0:
        params["add_nodes"] = int(add_nodes)
    r = requests.post(STRING_API, data=params, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    if df.empty or "preferredName_A" not in df.columns:
        return pd.DataFrame(columns=["source", "target", "weight"])
    out = df[["preferredName_A", "preferredName_B", "score"]].copy()
    out.columns = ["source", "target", "weight"]
    out["source"] = out["source"].astype(str).str.upper()
    out["target"] = out["target"].astype(str).str.upper()
    return out


def build_graph(edges, nodes=None, min_weight=0.0, drop_isolates=True, restrict_to=None):
    """Build an undirected weighted graph from an edge table."""
    G = nx.Graph()
    if nodes:
        G.add_nodes_from(nodes)
    allowed = set(restrict_to) if restrict_to else None
    for s, t, w in edges[["source", "target", "weight"]].itertuples(index=False):
        if s == t or pd.isna(w) or w < min_weight:
            continue
        if allowed is not None and (s not in allowed or t not in allowed):
            continue
        w = float(w)
        if G.has_edge(s, t):
            w = max(w, G[s][t]["weight"])
        G.add_edge(s, t, weight=w)
    if drop_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))
    return G


def compute_metrics(G, drug_set, disease_set):
    """Centrality metrics, community ids and gene-origin labels for every node."""
    nodes = list(G.nodes())
    deg = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    betw = nx.betweenness_centrality(G)
    clos = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=2000, weight="weight")
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eig = {n: 0.0 for n in nodes}
    pr = nx.pagerank(G, weight="weight")

    comm_id = {}
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        for i, c in enumerate(comms):
            for n in c:
                comm_id[n] = i + 1
    except Exception:
        pass

    def origin(g):
        in_d, in_s = g in drug_set, g in disease_set
        if in_d and in_s:
            return "Drug ∩ Disease"
        if in_d:
            return "Drug only"
        if in_s:
            return "Disease only"
        return "Added interactor"

    return pd.DataFrame(
        {
            "gene": nodes,
            "origin": [origin(n) for n in nodes],
            "degree": [deg[n] for n in nodes],
            "weighted_degree": [round(strength[n], 4) for n in nodes],
            "betweenness": [betw[n] for n in nodes],
            "closeness": [clos[n] for n in nodes],
            "eigenvector": [eig.get(n, 0.0) for n in nodes],
            "pagerank": [pr[n] for n in nodes],
            "community": [comm_id.get(n, 0) for n in nodes],
        }
    )


def rank_targets(metrics, chosen):
    """Rank genes by the mean of their ranks across the chosen centrality metrics."""
    out = metrics.copy()
    ranks = pd.concat(
        [out[m].rank(ascending=False, method="min") for m in chosen], axis=1
    )
    out["mean_rank"] = ranks.mean(axis=1)
    out = out.sort_values(["mean_rank", "degree"], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def draw_network(G, meta, color_by="Gene origin", label_genes=None, max_auto_labels=45):
    """Spring-layout network. Node size = degree; colour = origin or community."""
    fig, ax = plt.subplots(figsize=(9, 7))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes to display", ha="center", va="center")
        ax.axis("off")
        return fig

    n = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.6 / np.sqrt(n))
    deg = np.array([G.degree(x) for x in G.nodes()], dtype=float)
    sizes = 120 + 1400 * (deg - deg.min()) / (deg.max() - deg.min() + 1e-9)

    w = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    wmax = w.max() if w.size and w.max() > 0 else 1.0
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=0.3 + 2.2 * (w / wmax), alpha=0.35, edge_color="#888888"
    )

    if color_by == "Community":
        comm = [int(meta.loc[x, "community"]) % 20 for x in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=sizes, node_color=comm, cmap=plt.cm.tab20,
            vmin=0, vmax=19, edgecolors="white", linewidths=0.8,
        )
    else:
        origins = [meta.loc[x, "origin"] for x in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=sizes,
            node_color=[ORIGIN_COLORS[o] for o in origins],
            edgecolors="white", linewidths=0.8,
        )
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=o)
            for o, c in ORIGIN_COLORS.items() if o in set(origins)
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    if label_genes is None:
        label_genes = set(G.nodes()) if n <= max_auto_labels else set()
    labels = {g: g for g in G.nodes() if g in label_genes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ======================================================================
# UI
# ======================================================================

st.set_page_config(page_title="Target Overlap & PPI Network", layout="wide")
st.title("🧬 Target Overlap & PPI Network")
st.caption(
    "Find genes shared between a drug's targets and a disease, build a protein–protein "
    "interaction network, and rank the top interacting targets."
)

for k, v in {
    "tn_drug_text": "",
    "tn_disease_text": "",
    "tn_drug_label": "Drug",
    "tn_disease_label": "Disease",
    "tn_graph": None,
    "tn_metrics": None,
    "tn_edges": None,
    "tn_geneset_name": None,
}.items():
    st.session_state.setdefault(k, v)


def _load_sample():
    st.session_state["tn_drug_text"] = SAMPLE_DRUG_GENES
    st.session_state["tn_disease_text"] = SAMPLE_DISEASE_GENES
    st.session_state["tn_drug_label"] = "Drug (sample)"
    st.session_state["tn_disease_label"] = "Disease (sample)"


def read_gene_file(uploaded, key):
    """Read genes from a .txt (one token per gene) or .csv/.tsv (pick a column)."""
    raw = uploaded.getvalue()
    if uploaded.name.lower().endswith(".txt"):
        return parse_text_genes(raw.decode("utf-8", errors="ignore"))
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    if df.shape[1] == 1:
        col = df.columns[0]
    else:
        col = st.selectbox("Gene column", df.columns.tolist(), key=f"{key}_col")
    return clean_gene_list(df[col].tolist())


def gather_genes(text, uploaded, key):
    genes = parse_text_genes(text)
    if uploaded is not None:
        genes = clean_gene_list(genes + read_gene_file(uploaded, key))
    return genes


tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Gene Lists", "2. Venn Diagram", "3. PPI Network", "4. Top Targets"]
)

# ----------------------------------------------------------------------
# TAB 1 — Gene lists
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Provide your gene lists")
    st.write(
        "Use official gene symbols (e.g. `PPARG`, `TNF`). Paste them in, upload a file, or both. "
        "Symbols are upper-cased and de-duplicated automatically."
    )
    st.button("🧪 Load sample gene lists", on_click=_load_sample)
    st.caption(
        "The sample lists are illustrative only (a generic metabolic-disease example), "
        "not curated from a database."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Drug name", key="tn_drug_label")
        st.text_area("Drug target genes", key="tn_drug_text", height=220,
                     help="One per line, or separated by commas / spaces.")
        f1 = st.file_uploader("…or upload .txt / .csv", type=["txt", "csv", "tsv"], key="tn_drug_file")
        drug_genes = gather_genes(st.session_state["tn_drug_text"], f1, "tn_drug")
        st.write(f"**{len(drug_genes)}** unique genes")
    with c2:
        st.text_input("Disease name", key="tn_disease_label")
        st.text_area("Disease genes", key="tn_disease_text", height=220,
                     help="One per line, or separated by commas / spaces.")
        f2 = st.file_uploader("…or upload .txt / .csv", type=["txt", "csv", "tsv"], key="tn_disease_file")
        disease_genes = gather_genes(st.session_state["tn_disease_text"], f2, "tn_disease")
        st.write(f"**{len(disease_genes)}** unique genes")

drug_label = st.session_state["tn_drug_label"] or "Drug"
disease_label = st.session_state["tn_disease_label"] or "Disease"
A, B = set(drug_genes), set(disease_genes)

# ----------------------------------------------------------------------
# TAB 2 — Venn diagram
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Overlapping genes")
    if not A or not B:
        st.info("Add genes for both the drug and the disease in Tab 1 first.")
    else:
        bg = st.number_input(
            "Background gene universe size (for the enrichment test)",
            min_value=1000, max_value=60000, value=20000, step=1000,
            help="~20,000 is a common approximation for human protein-coding genes.",
        )
        stats = overlap_stats(A, B, bg)

        left, right = st.columns([3, 2])
        with left:
            fig = draw_venn(A, B, drug_label, disease_label)
            st.pyplot(fig)
            png = fig_to_png_bytes(fig)
            plt.close(fig)
            st.download_button("Download Venn (PNG, 300 dpi)", png, "venn_diagram.png", "image/png")
        with right:
            st.metric("Overlapping genes", stats["overlap"])
            m1, m2 = st.columns(2)
            m1.metric("Jaccard index", f"{stats['jaccard']:.3f}")
            m2.metric("Expected by chance", f"{stats['expected']:.2f}")
            st.metric("Hypergeometric p-value", f"{stats['p_value']:.2e}")
            st.caption(
                "Probability of seeing at least this many shared genes if both lists were "
                f"drawn at random from {stats['background']:,} genes."
            )

        overlap_list, a_only, b_only = sorted(A & B), sorted(A - B), sorted(B - A)
        c1, c2, c3 = st.columns(3)
        for col, title, items, fname in [
            (c1, f"{drug_label} only", a_only, "drug_only_genes.csv"),
            (c2, "Overlap", overlap_list, "overlapping_genes.csv"),
            (c3, f"{disease_label} only", b_only, "disease_only_genes.csv"),
        ]:
            with col:
                st.write(f"**{title}** ({len(items)})")
                st.dataframe(pd.DataFrame({"gene": items}), height=260)
                st.download_button(f"Download {title}", to_csv_bytes(pd.DataFrame({"gene": items})),
                                   fname, "text/csv", key=f"dl_{fname}")

# ----------------------------------------------------------------------
# TAB 3 — PPI network
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Protein–protein interaction network")
    if not A or not B:
        st.info("Add genes for both the drug and the disease in Tab 1 first.")
    else:
        gene_sets = {
            "Overlap (drug ∩ disease)": sorted(A & B),
            "Union (drug ∪ disease)": sorted(A | B),
            f"{drug_label} only": sorted(A - B),
            f"{disease_label} only": sorted(B - A),
        }
        set_name = st.radio("Genes to build the network from", list(gene_sets.keys()), horizontal=True)
        net_genes = gene_sets[set_name]
        st.write(f"**{len(net_genes)}** genes selected")
        if len(net_genes) < 2:
            st.warning("Pick a gene set with at least 2 genes to build a network.")

        source = st.radio(
            "Interaction source",
            ["Fetch from STRING (online)", "Upload my own interaction table"],
            horizontal=True,
        )

        edges, min_weight, restrict, add_nodes = None, 0.0, True, 0
        if source.startswith("Fetch"):
            c1, c2, c3 = st.columns(3)
            with c1:
                species = st.selectbox(
                    "Species",
                    [9606, 10090, 10116],
                    format_func=lambda x: {9606: "Human (9606)", 10090: "Mouse (10090)", 10116: "Rat (10116)"}[x],
                )
            with c2:
                score = st.select_slider(
                    "Minimum interaction confidence",
                    options=[150, 400, 700, 900],
                    value=400,
                    format_func=lambda x: {150: "Low (0.15)", 400: "Medium (0.4)", 700: "High (0.7)", 900: "Highest (0.9)"}[x],
                )
            with c3:
                add_nodes = st.slider("Add extra interactors (STRING)", 0, 50, 0,
                                      help="Asks STRING to add this many additional highly connected "
                                           "proteins to the network.")
            st.caption("This calls string-db.org from the machine running the app, so it needs internet access.")
            uploaded_edges = None
        else:
            uploaded_edges = st.file_uploader(
                "Interaction table (.csv / .tsv / .txt) — e.g. a STRING or BioGRID export",
                type=["csv", "tsv", "txt"], key="tn_edges_file",
            )
            if uploaded_edges is not None:
                raw_edges = pd.read_csv(io.BytesIO(uploaded_edges.getvalue()), sep=None, engine="python")
                st.dataframe(raw_edges.head(8))
                cols = raw_edges.columns.tolist()

                def guess(cands, default=0):
                    low = {c.lower(): i for i, c in enumerate(cols)}
                    for c in cands:
                        if c.lower() in low:
                            return low[c.lower()]
                    return default

                c1, c2, c3 = st.columns(3)
                with c1:
                    s_col = st.selectbox("Source protein column", cols,
                                         index=guess(["preferredName_A", "node1", "protein1", "source", "gene1", "from"], 0))
                with c2:
                    t_col = st.selectbox("Target protein column", cols,
                                         index=guess(["preferredName_B", "node2", "protein2", "target", "gene2", "to"], min(1, len(cols) - 1)))
                with c3:
                    w_options = ["(none — treat all as 1)"] + cols
                    w_guess = guess(["score", "combined_score", "weight", "confidence"], -1)
                    w_col = st.selectbox("Weight / score column (optional)", w_options,
                                         index=w_guess + 1 if w_guess >= 0 else 0)
                min_weight = st.number_input("Minimum weight to keep an edge", value=0.0,
                                             help="STRING exports use 0–1 (or 0–1000 for combined_score).")
                restrict = st.checkbox("Keep only edges between the selected genes", value=True)

                edges = pd.DataFrame(
                    {
                        "source": raw_edges[s_col].astype(str).str.upper().str.strip(),
                        "target": raw_edges[t_col].astype(str).str.upper().str.strip(),
                        "weight": 1.0 if w_col.startswith("(none") else pd.to_numeric(raw_edges[w_col], errors="coerce"),
                    }
                )

        drop_iso = st.checkbox("Remove genes with no interactions from the network", value=True)

        if st.button("Build network & compute metrics", type="primary", disabled=len(net_genes) < 2):
            try:
                if source.startswith("Fetch"):
                    with st.spinner("Querying STRING…"):
                        edges = fetch_string_network(tuple(net_genes), species, score, add_nodes)
                    restrict_to = None if add_nodes > 0 else net_genes
                elif edges is None:
                    raise ValueError("Upload an interaction table first.")
                else:
                    restrict_to = net_genes if restrict else None

                G = build_graph(edges, nodes=net_genes, min_weight=min_weight,
                                drop_isolates=drop_iso, restrict_to=restrict_to)
                if G.number_of_edges() == 0:
                    st.session_state["tn_graph"] = None
                    st.session_state["tn_metrics"] = None
                    st.warning(
                        "No interactions were found for these genes at this threshold. "
                        "Try a lower confidence cut-off, a larger gene set (e.g. the union), "
                        "or check that the gene symbols match the species."
                    )
                else:
                    st.session_state["tn_graph"] = G
                    st.session_state["tn_edges"] = edges
                    st.session_state["tn_metrics"] = compute_metrics(G, A, B)
                    st.session_state["tn_geneset_name"] = set_name
            except requests.exceptions.RequestException as e:
                st.error(
                    "Could not reach STRING. Check your internet connection, or switch to "
                    f"'Upload my own interaction table'.\n\nDetails: {e}"
                )
            except Exception as e:
                st.error(f"Could not build the network: {e}")

        G = st.session_state["tn_graph"]
        metrics = st.session_state["tn_metrics"]
        if G is not None and metrics is not None:
            st.success(f"Network built from: {st.session_state['tn_geneset_name']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nodes", G.number_of_nodes())
            m2.metric("Edges", G.number_of_edges())
            m3.metric("Density", f"{nx.density(G):.3f}")
            m4.metric("Connected components", nx.number_connected_components(G))

            color_by = st.radio("Colour nodes by", ["Gene origin", "Community"], horizontal=True)
            if G.number_of_nodes() > 300:
                st.warning("Large network — drawing may be slow and unreadable. Consider a higher confidence cut-off.")
            fig = draw_network(G, metrics.set_index("gene"), color_by)
            st.pyplot(fig)
            png = fig_to_png_bytes(fig)
            plt.close(fig)
            st.download_button("Download network (PNG, 300 dpi)", png, "ppi_network.png", "image/png")
            st.caption("Node size = degree (number of interaction partners). Edge thickness = interaction confidence.")

# ----------------------------------------------------------------------
# TAB 4 — Top targets
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Top interacting targets")
    G = st.session_state["tn_graph"]
    metrics = st.session_state["tn_metrics"]
    if G is None or metrics is None:
        st.info("Build a network in Tab 3 first.")
    else:
        n_nodes = len(metrics)
        c1, c2 = st.columns([2, 1])
        with c1:
            chosen_labels = st.multiselect(
                "Metrics combined into the overall ranking",
                list(METRIC_LABELS.keys()),
                default=["Degree", "Betweenness", "Closeness", "Eigenvector"],
                help="Each gene is ranked on every chosen metric; the overall rank uses the mean of those ranks.",
            )
        with c2:
            top_n = st.slider("Top N targets", 1, n_nodes, min(10, n_nodes))

        if not chosen_labels:
            st.warning("Select at least one metric.")
        else:
            chosen = [METRIC_LABELS[x] for x in chosen_labels]
            ranked = rank_targets(metrics, chosen)
            top = ranked.head(top_n)

            st.write(f"**Top {top_n} targets** (lower mean rank = more central)")
            st.dataframe(
                top.style.format(
                    {"betweenness": "{:.4f}", "closeness": "{:.4f}", "eigenvector": "{:.4f}",
                     "pagerank": "{:.4f}", "mean_rank": "{:.1f}"}
                )
            )

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download full ranking (CSV)", to_csv_bytes(ranked),
                                   "ranked_targets.csv", "text/csv")
            with d2:
                edge_df = nx.to_pandas_edgelist(G)
                st.download_button("Download network edges (CSV)", to_csv_bytes(edge_df),
                                   "ppi_edges.csv", "text/csv")

            left, right = st.columns(2)
            with left:
                bar_label = st.selectbox("Plot metric", list(METRIC_LABELS.keys()), index=0)
                bar_col = METRIC_LABELS[bar_label]
                bar_df = ranked.sort_values(bar_col, ascending=False).head(top_n)
                figb, axb = plt.subplots(figsize=(6, max(3, 0.35 * top_n + 1)))
                axb.barh(bar_df["gene"][::-1], bar_df[bar_col][::-1],
                         color=[ORIGIN_COLORS[o] for o in bar_df["origin"][::-1]])
                axb.set_xlabel(bar_label)
                axb.legend(
                    handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=o)
                             for o, c in ORIGIN_COLORS.items() if o in set(bar_df["origin"])],
                    frameon=False, fontsize=8, loc="lower right",
                )
                figb.tight_layout()
                st.pyplot(figb)
                plt.close(figb)
            with right:
                st.write("**Sub-network of the top targets**")
                top_genes = set(top["gene"])
                figs = draw_network(G.subgraph(top_genes).copy(), metrics.set_index("gene"),
                                    "Gene origin", label_genes=top_genes)
                st.pyplot(figs)
                plt.close(figs)

st.sidebar.header("About")
st.sidebar.write(
    "Compare a drug's target genes with disease genes, build a PPI network, and rank "
    "hub targets by network centrality."
)
st.sidebar.write(
    "PPI data are retrieved from the [STRING database](https://string-db.org) "
    "or from your own interaction table."
)
