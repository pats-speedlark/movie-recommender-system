"""CineMatch - a content-based movie recommender built with Streamlit."""

from __future__ import annotations

import html
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional in production
    pass

DATA_DIR = Path(__file__).resolve().parent / "data"
TMDB_API = "https://api.themoviedb.org/3/movie/{movie_id}"
TMDB_IMG = "https://image.tmdb.org/t/p/w342"
TMDB_PAGE = "https://www.themoviedb.org/movie/{movie_id}"
PLACEHOLDER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='342' height='513' viewBox='0 0 342 513'>"
    "<rect width='100%25' height='100%25' fill='%23222631'/>"
    "<text x='50%25' y='50%25' fill='%23737b8c' font-family='sans-serif' font-size='22' "
    "text-anchor='middle'>No poster</text></svg>"
)

st.set_page_config(page_title="CineMatch · Movie Recommender", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    key = os.getenv("TMDB_API_KEY") or os.getenv("API_KEY")
    if key:
        return key.strip()
    try:
        return st.secrets.get("TMDB_API_KEY") or st.secrets.get("API_KEY")
    except Exception:  # no secrets.toml present
        return None


API_KEY = get_api_key()


# --------------------------------------------------------------------------- #
# Data / model (loaded once per process, shared across sessions)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading movie catalogue…")
def load_model() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    movies = pd.read_csv(DATA_DIR / "movies.csv", usecols=["movie_id", "title"])
    neighbors = np.load(DATA_DIR / "neighbors.npz")
    indices = neighbors["indices"]
    scores = neighbors["scores"].astype(np.float32)

    # Make labels unique so duplicate titles (e.g. two "Batman" films) are selectable.
    dup = movies["title"].duplicated(keep=False)
    movies["label"] = movies["title"]
    movies.loc[dup, "label"] = movies.loc[dup, "title"] + " (#" + movies.loc[dup, "movie_id"].astype(str) + ")"
    return movies, indices, scores


@st.cache_resource
def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=20))
    return session


@st.cache_data(ttl=60 * 60 * 24, max_entries=5000, show_spinner=False)
def fetch_details(movie_id: int) -> dict:
    """Fetch poster/rating/year/overview from TMDB. Never raises."""
    details = {"poster": PLACEHOLDER, "rating": None, "year": None, "overview": "", "genres": []}
    if not API_KEY:
        return details
    try:
        resp = http_session().get(
            TMDB_API.format(movie_id=movie_id), params={"api_key": API_KEY}, timeout=5
        )
        if resp.status_code != 200:
            return details
        data = resp.json()
    except (requests.RequestException, ValueError):
        return details

    if data.get("poster_path"):
        details["poster"] = TMDB_IMG + data["poster_path"]
    if data.get("vote_average"):
        details["rating"] = round(float(data["vote_average"]), 1)
    if data.get("release_date"):
        details["year"] = data["release_date"][:4]
    details["overview"] = data.get("overview") or ""
    details["genres"] = [g["name"] for g in data.get("genres", [])][:3]
    return details


def fetch_many(movie_ids: list[int]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=10) as pool:
        return list(pool.map(fetch_details, movie_ids))


# --------------------------------------------------------------------------- #
# Recommendation logic
# --------------------------------------------------------------------------- #
def recommend(seed_rows: list[int], n: int) -> list[tuple[int, float]]:
    """Return (row index, score) pairs.

    With one seed this is plain nearest-neighbour lookup. With several seeds the
    neighbour scores are summed, so films similar to *all* picks rank highest.
    """
    _, indices, scores = load_model()
    seeds = set(seed_rows)
    agg: dict[int, float] = {}
    for row in seed_rows:
        for j, s in zip(indices[row], scores[row]):
            j = int(j)
            if j not in seeds:
                agg[j] = agg.get(j, 0.0) + float(s)
    # Mean similarity across seeds (missing neighbours count as 0).
    k = len(seed_rows)
    ranked = sorted(((j, total / k) for j, total in agg.items()), key=lambda x: -x[1])
    return ranked[:n]


def match_percent(score: float, best: float) -> int:
    """Map raw cosine similarity to a friendly 0-100 relative match score."""
    if best <= 0:
        return 0
    return int(round(55 + 44 * (score / best)))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero h1 {font-size: 2.6rem; margin-bottom: 0;}
.hero h1 .brand {background: linear-gradient(90deg,#ff4b6e,#ffb84b);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {color: #9aa3b5; margin-top: .25rem; font-size: 1.05rem;}
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 1.25rem; margin-top: 1rem;}
.card {background: #1a1d26; border-radius: 14px; overflow: hidden; position: relative;
  box-shadow: 0 4px 18px rgba(0,0,0,.35); transition: transform .18s ease, box-shadow .18s ease;}
.card:hover {transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,0,0,.5);}
.card a {text-decoration: none; color: inherit;}
.card img {width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block; background: #222631;}
.card .body {padding: .6rem .75rem .8rem;}
.card .title {font-weight: 600; font-size: .95rem; line-height: 1.25; color: #f1f3f8;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
.card .meta {font-size: .78rem; color: #9aa3b5; margin-top: .3rem;}
.card .badge {position: absolute; top: .5rem; left: .5rem; background: rgba(15,17,23,.85);
  color: #7ee2a8; font-weight: 700; font-size: .75rem; padding: .2rem .5rem; border-radius: 999px;}
.card .rating {position: absolute; top: .5rem; right: .5rem; background: rgba(15,17,23,.85);
  color: #ffd166; font-weight: 700; font-size: .75rem; padding: .2rem .5rem; border-radius: 999px;}
.card .overview {position: absolute; inset: 0; background: rgba(12,14,20,.93); color: #d6dae3;
  font-size: .8rem; line-height: 1.35; padding: .9rem; opacity: 0; transition: opacity .2s ease;
  overflow: hidden;}
.card:hover .overview {opacity: 1;}
.seed {display: flex; gap: 1.25rem; align-items: flex-start; background: #1a1d26; padding: 1rem;
  border-radius: 14px; margin-bottom: .5rem;}
.seed img {width: 120px; border-radius: 10px;}
.seed h3 {margin: 0 0 .25rem 0;}
.seed .meta {color: #9aa3b5; font-size: .9rem; margin-bottom: .5rem;}
.seed p {color: #d6dae3; font-size: .92rem; margin: 0;}
@media (max-width: 640px) {
  .grid {grid-template-columns: repeat(2, 1fr); gap: .8rem;}
  .hero h1 {font-size: 2rem;}
  .seed img {width: 90px;}
}
</style>
"""


def flat(markup: str) -> str:
    """Collapse indented HTML onto one line so Markdown doesn't treat it as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


def meta_line(d: dict) -> str:
    parts = [p for p in [d.get("year"), " · ".join(d.get("genres") or [])] if p]
    return html.escape(" · ".join(parts))


def render_seed(title: str, movie_id: int, d: dict) -> None:
    st.markdown(
        flat(f"""<div class="seed">
              <img src="{html.escape(d['poster'])}" alt="poster"/>
              <div>
                <h3>{html.escape(title)}</h3>
                <div class="meta">{meta_line(d)}{' · ⭐ ' + str(d['rating']) if d.get('rating') else ''}</div>
                <p>{html.escape(d.get('overview') or '')}</p>
              </div>
            </div>"""),
        unsafe_allow_html=True,
    )


def render_grid(items: list[tuple[str, int, int, dict]]) -> None:
    cards = []
    for title, movie_id, pct, d in items:
        rating = f'<span class="rating">⭐ {d["rating"]}</span>' if d.get("rating") else ""
        overview = html.escape(d.get("overview") or "")
        cards.append(
            flat(f"""<div class="card"><a href="{TMDB_PAGE.format(movie_id=movie_id)}" target="_blank" rel="noopener">
                  <img src="{html.escape(d['poster'])}" alt="{html.escape(title)}" loading="lazy"/>
                  <span class="badge">{pct}% match</span>{rating}
                  {f'<div class="overview">{overview}</div>' if overview else ''}
                  <div class="body"><div class="title">{html.escape(title)}</div>
                  <div class="meta">{meta_line(d)}</div></div>
                </a></div>""")
        )
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    movies, _, _ = load_model()
    labels = movies["label"].tolist()
    label_to_row = {label: i for i, label in enumerate(labels)}

    st.markdown(
        '<div class="hero"><h1>🎬 <span class="brand">CineMatch</span></h1>'
        f"<p>Pick up to five movies you love and get similar picks from {len(movies):,} films — "
        "matched on plot, genres, keywords, cast and director.</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        n_recs = st.slider("Number of recommendations", 5, 30, 10, step=5)
        st.caption(
            "Recommendations use TF-IDF similarity over each film's plot, genres, "
            "keywords, top cast and director."
        )
        if not API_KEY:
            st.warning(
                "No TMDB API key found — posters and ratings are disabled. "
                "Set the `TMDB_API_KEY` (or `API_KEY`) environment variable."
            )

    if "picks" not in st.session_state:
        st.session_state.picks = [labels[0]]

    def surprise() -> None:
        st.session_state.picks = [random.choice(labels)]

    col_sel, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_sel:
        picks = st.multiselect(
            "Search for one or more movies you like",
            options=labels,
            key="picks",
            max_selections=5,
            placeholder="Start typing a movie title…",
        )
    with col_btn:
        st.button("🎲 Surprise me", on_click=surprise, use_container_width=True)

    if not picks:
        st.info("Select at least one movie to get recommendations.")
        return

    seed_rows = [label_to_row[p] for p in picks]
    results = recommend(seed_rows, n_recs)
    if not results:
        st.warning("No recommendations found for that selection.")
        return

    rec_ids = [int(movies.at[j, "movie_id"]) for j, _ in results]
    seed_ids = [int(movies.at[r, "movie_id"]) for r in seed_rows]
    with st.spinner("Fetching posters…"):
        details = fetch_many(seed_ids + rec_ids)
    seed_details, rec_details = details[: len(seed_ids)], details[len(seed_ids):]

    if len(seed_rows) == 1:
        render_seed(movies.at[seed_rows[0], "title"], seed_ids[0], seed_details[0])

    heading = "Because you liked " + ", ".join(f"**{movies.at[r, 'title']}**" for r in seed_rows)
    st.markdown(f"### {heading}")

    best = results[0][1]
    render_grid(
        [
            (movies.at[j, "title"], rec_ids[i], match_percent(score, best), rec_details[i])
            for i, (j, score) in enumerate(results)
        ]
    )

    st.markdown("---")
    st.caption("Data & images courtesy of TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")


main()
