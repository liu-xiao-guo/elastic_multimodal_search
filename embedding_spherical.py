import numpy as np
import webbrowser, tempfile, os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.graph_objects as go

WORDS = ["cat", "dog", "kitten", "Puppy","king", "queen", "猫", "狗", "table", "chair", "pizza", "pasta", "asymptomatic"]
MODEL_NAME = "jinaai/jina-embeddings-v4"

CATEGORY_COLORS = {
    "animals":   "#e74c3c",  # red
    "royalty":   "#f39c12",  # orange
    "furniture": "#3498db",  # blue
    "food":      "#2ecc71",  # green
    "medical":   "#9b59b6",  # purple
    "other":     "#95a5a6",  # grey
}
WORD_CATEGORIES = {
    "cat": "animals", "dog": "animals", "kitten": "animals", "Puppy": "animals",
    "猫": "animals", "狗": "animals",
    "king": "royalty", "queen": "royalty",
    "table": "furniture", "chair": "furniture",
    "pizza": "food", "pasta": "food",
    "asymptomatic": "medical",
}
COLORS = [CATEGORY_COLORS[WORD_CATEGORIES.get(w, "other")] for w in WORDS]

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

print("Computing embeddings...")
embeddings = model.encode(WORDS, task="retrieval", normalize_embeddings=True)
print(f"Embedding shape: {embeddings.shape}")

# Angular (great-circle) distance between unit embeddings in the original space.
# Since the vectors are unit length, this is the true geodesic distance on the
# embedding hypersphere: arccos(cosine similarity).
cos_sim = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
ang_dist = np.arccos(cos_sim)  # radians

# Project to 3D with PCA (centered so points spread over the sphere), then push
# every point onto the unit sphere.
pca3 = PCA(n_components=3)
coords3 = pca3.fit_transform(embeddings)
exp3 = pca3.explained_variance_ratio_ * 100
print(f"3D PCA explained variance: {exp3[0]:.1f}%, {exp3[1]:.1f}%, {exp3[2]:.1f}%")

norms = np.linalg.norm(coords3, axis=1, keepdims=True)
sphere = coords3 / np.where(norms == 0, 1, norms)

# Distance measured on the 3D unit sphere (what you actually see).
shown_dist = np.arccos(np.clip(sphere @ sphere.T, -1.0, 1.0))


def slerp(a, b, n=40):
    """Points along the great-circle arc (shortest path) from a to b."""
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    t = np.linspace(0, 1, n)[:, None]
    if omega < 1e-8:
        return np.repeat(a[None, :], n, axis=0)
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)


def sphere_surface():
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    return (np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)))


def grid_lines(step_deg=30, n=100):
    """Latitude / longitude lines as a single polyline trace (None separates segments)."""
    xs, ys, zs = [], [], []
    t = np.linspace(0, 2 * np.pi, n)
    for lat in np.arange(-60, 61, step_deg):
        r, z = np.cos(np.radians(lat)), np.sin(np.radians(lat))
        xs += list(r * np.cos(t)) + [None]
        ys += list(r * np.sin(t)) + [None]
        zs += [z] * n + [None]
    p = np.linspace(0, np.pi, n)
    for lon in np.arange(0, 180, step_deg):
        l = np.radians(lon)
        xs += list(np.sin(p) * np.cos(l)) + [None]
        ys += list(np.sin(p) * np.sin(l)) + [None]
        zs += list(np.cos(p)) + [None]
    return xs, ys, zs


fig = go.Figure()

# Sphere body + grid
sx, sy, sz = sphere_surface()
fig.add_trace(go.Surface(
    x=sx, y=sy, z=sz, showscale=False, opacity=0.12,
    colorscale=[[0, "#cfd8e6"], [1, "#cfd8e6"]],
    hoverinfo="skip", showlegend=False, lighting=dict(ambient=1.0, diffuse=0.0),
))
gx, gy, gz = grid_lines()
fig.add_trace(go.Scatter3d(
    x=gx, y=gy, z=gz, mode="lines", showlegend=False, hoverinfo="skip",
    line=dict(color="#c5cdd9", width=1),
))

# Word markers (traces 2 .. 2+len(WORDS)-1)
label_traces = []
for i, word in enumerate(WORDS):
    label_traces.append(len(fig.data))
    fig.add_trace(go.Scatter3d(
        x=[sphere[i, 0]], y=[sphere[i, 1]], z=[sphere[i, 2]],
        mode="markers+text", name=word, text=[word],
        textposition="top center",
        textfont=dict(size=13, color=COLORS[i]),
        marker=dict(size=8, color=COLORS[i], opacity=1.0,
                    line=dict(width=1, color="white")),
        hovertemplate=(
            f"<b>{word}</b><br>"
            f"x: {sphere[i,0]:.4f}<br>"
            f"y: {sphere[i,1]:.4f}<br>"
            f"z: {sphere[i,2]:.4f}<extra></extra>"
        ),
    ))

# Great-circle arcs between every pair; hover shows the geodesic distance.
# Arcs sit marginally above the surface so they are not hidden by it.
arc_traces = []
for i in range(len(WORDS)):
    for j in range(i + 1, len(WORDS)):
        arc = slerp(sphere[i], sphere[j]) * 1.004
        arc_traces.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=arc[:, 0], y=arc[:, 1], z=arc[:, 2],
            mode="lines", showlegend=False,
            line=dict(color="rgba(120,130,150,0.35)", width=2),
            hovertemplate=(
                f"{WORDS[i]} <-> {WORDS[j]}<br>"
                f"geodesic (embedding): {ang_dist[i, j]:.4f} rad "
                f"({np.degrees(ang_dist[i, j]):.1f}°)<br>"
                f"arc on sphere: {shown_dist[i, j]:.4f} rad "
                f"({np.degrees(shown_dist[i, j]):.1f}°)<br>"
                f"cosine sim: {cos_sim[i, j]:.4f}<extra></extra>"
            ),
        ))

# Highlight each word's nearest-neighbour arc and print its distance at the midpoint.
nn_pairs = set()
for i in range(len(WORDS)):
    d = ang_dist[i].copy()
    d[i] = np.inf
    j = int(np.argmin(d))
    nn_pairs.add((min(i, j), max(i, j)))

nn_traces = []
for i, j in sorted(nn_pairs):
    arc = slerp(sphere[i], sphere[j]) * 1.006
    mid = arc[len(arc) // 2]
    nn_traces.append(len(fig.data))
    fig.add_trace(go.Scatter3d(
        x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines", showlegend=False,
        line=dict(color="#2c3e50", width=5),
        hovertemplate=(f"{WORDS[i]} <-> {WORDS[j]} (nearest)<br>"
                       f"geodesic: {np.degrees(ang_dist[i, j]):.1f}°<extra></extra>"),
    ))
    nn_traces.append(len(fig.data))
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]], mode="text", showlegend=False,
        text=[f"{np.degrees(ang_dist[i, j]):.1f}°"],
        textfont=dict(size=11, color="#2c3e50"),
        hoverinfo="skip",
    ))


def vis_buttons(label_idx, arc_idx, nn_idx):
    return [
        dict(label="Show Labels", method="restyle",
             args=[{"mode": "markers+text"}, label_idx]),
        dict(label="Hide Labels", method="restyle",
             args=[{"mode": "markers"}, label_idx]),
        dict(label="All Arcs", method="restyle",
             args=[{"visible": True}, arc_idx + nn_idx]),
        dict(label="Nearest Only", method="restyle",
             args=[{"visible": False}, arc_idx]),
        dict(label="Hide Arcs", method="restyle",
             args=[{"visible": False}, arc_idx + nn_idx]),
    ]


fig.update_layout(
    title=dict(
        text=(f"Word Embeddings - Spherical View<br><sup>Model: {MODEL_NAME} | "
              f"PCA→3D, projected onto the unit sphere | Variance kept: {sum(exp3):.1f}% | "
              f"Arcs = great-circle (angular) distance</sup>"),
        x=0.5,
    ),
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.1, y=1.1, z=0.75)),
    ),
    legend=dict(title="Words", x=1.02, y=0.98, xanchor="left", yanchor="top"),
    margin=dict(l=0, r=140, t=80, b=0),
    width=1050, height=750,
    updatemenus=[dict(
        type="buttons", showactive=False, direction="down",
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        buttons=vis_buttons(label_traces, arc_traces, nn_traces),
    )],
)

out = os.path.join(tempfile.gettempdir(), "word_embeddings_spherical.html")
fig.write_html(out, include_plotlyjs="cdn")

print("\nGeodesic distance (degrees) to each word's nearest neighbour:")
for i, j in sorted(nn_pairs):
    print(f"  {WORDS[i]:>14} <-> {WORDS[j]:<14} {np.degrees(ang_dist[i, j]):6.1f}°")

print(f"Saved to {out}")
print("Opening in browser...")
webbrowser.open(f"file://{out}")
