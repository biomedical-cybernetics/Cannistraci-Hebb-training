#!/usr/bin/env python3
"""Create network-structure frames and animations from MLP adjacency .mat files.

This script is a cleaned, runnable version of create_video.ipynb. It keeps the
notebook's core behavior, but moves repeated paths and plotting steps into
explicit functions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence


LAYER_NAMES = ("1", "2", "3")
DEFAULT_EPOCHS = tuple(range(100))
DEFAULT_LAYER_SIZES = (784, 1568, 1568, 1568)
LAYER_RGB_VALUES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (238.0 / 255.0, 130.0 / 255.0, 238.0 / 255.0),
)
LAYER_HEX = ("#ff0000", "#00ff00", "#00ffff", "#ee82ee")
np = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    adjacency_root: Path
    fc_res_path: Path
    cht_res_path: Path
    plain_image_dir: Path
    frame_dir: Path
    gif_path: Path
    embedding_mat_dir: Path
    layer_sizes: tuple[int, int, int, int] = DEFAULT_LAYER_SIZES

    def resolve(self, root: Path) -> "ExperimentConfig":
        def resolve_path(path: Path) -> Path:
            return path if path.is_absolute() else root / path

        return replace(
            self,
            adjacency_root=resolve_path(self.adjacency_root),
            fc_res_path=resolve_path(self.fc_res_path),
            cht_res_path=resolve_path(self.cht_res_path),
            plain_image_dir=resolve_path(self.plain_image_dir),
            frame_dir=resolve_path(self.frame_dir),
            gif_path=resolve_path(self.gif_path),
            embedding_mat_dir=resolve_path(self.embedding_mat_dir),
        )


CONFIGS: dict[str, ExperimentConfig] = {
    "emnist": ExperimentConfig(
        name="EMNIST",
        adjacency_root=Path(
            "mlp/EMNIST/s_0_lr_0.025_e_100/"
            "s_0.99_i_1_az_False_z_0.3_im_swi_um_zero_df_0.9/"
            "scs_/chain_/linear_/d_2_/CH3_L3_soft_weight_magnitude_soft"
        ),
        fc_res_path=Path(
            "mlp/EMNIST/s_0_lr_0.025_e_100/"
            "s_0.99_i_1_az_False_im_kaiming_um_zero_df_0.9/"
            "d_2_/fc_weight_magnitude/res.mat"
        ),
        cht_res_path=Path(
            "mlp/EMNIST/s_0_lr_0.025_e_100/"
            "s_0.99_i_1_az_False_z_0.3_im_swi_um_zero_df_0.9/"
            "scs_/chain_/linear_/d_2_/CH3_L3_soft_weight_magnitude_soft/res.mat"
        ),
        plain_image_dir=Path("hyperbolicity/EMNIST"),
        frame_dir=Path("hyperbolicity/Video"),
        gif_path=Path("hyperbolicity/image_animation.gif"),
        embedding_mat_dir=Path("hyperbolicity/EMNIST_mat"),
    ),
    "mnist_soft": ExperimentConfig(
        name="MNIST",
        adjacency_root=Path(
            "/mnt/ssd/yingtao/Cannistraci-Hebb Training/mlp/MNIST/"
            "s_0_lr_0.025_e_100/"
            "s_0.999_i_1_az_False_z_0.3_im_swi_um_zero_df_0.9/"
            "ws_beta_0.25/chain_/linear_/d_2_/CH3_L3_soft_weight_magnitude_soft"
        ),
        fc_res_path=Path(
            "mlp/MNIST/s_0_lr_0.025_e_100/"
            "s_0.99_i_1_az_False_im_kaiming_um_zero_df_0.9/"
            "d_2_/fc_weight_magnitude/res.mat"
        ),
        cht_res_path=Path(
            "/mnt/ssd/yingtao/Cannistraci-Hebb Training/mlp/MNIST/"
            "s_0_lr_0.025_e_100/"
            "s_0.999_i_1_az_False_z_0.3_im_swi_um_zero_df_0.9/"
            "ws_beta_0.25/chain_/linear_/d_2_/CH3_L3_soft_weight_magnitude_soft/res.mat"
        ),
        plain_image_dir=Path("hyperbolicity/MNIST_0.999"),
        frame_dir=Path("hyperbolicity/MNIST_0.999_Video"),
        gif_path=Path("hyperbolicity/MNIST_0.999_image_animation.gif"),
        embedding_mat_dir=Path("hyperbolicity/MNIST_0.999_mat"),
    ),
    "mnist_wm": ExperimentConfig(
        name="MNIST",
        adjacency_root=Path(
            "/mnt/ssd/yingtao/Cannistraci-Hebb Training/mlp/MNIST/"
            "s_0_lr_0.025_e_100/"
            "s_0.999_i_1_az_False_z_0.3_im_swi_um_zero_df_0.9/"
            "ws_beta_0.25/chain_/d_2_/CH3_L3_soft_weight_magnitude"
        ),
        fc_res_path=Path(
            "mlp/MNIST/s_0_lr_0.025_e_100/"
            "s_0.99_i_1_az_False_im_kaiming_um_zero_df_0.9/"
            "d_2_/fc_weight_magnitude/res.mat"
        ),
        cht_res_path=Path(
            "../Cannistraci-Hebb Training/mlp/MNIST/"
            "s_2_lr_0.025_e_100/"
            "s_0.999_i_1_az_False_z_0.3_im_swi_um_zero_df_0.9/"
            "ws_beta_0.25/chain_/d_2_/CH3_L3_soft_weight_magnitude/res.mat"
        ),
        plain_image_dir=Path("hyperbolicity/MNIST_0.999_WM"),
        frame_dir=Path("hyperbolicity/MNIST_0.999_Video_WM"),
        gif_path=Path("hyperbolicity/MNIST_0.999_WM_image_animation.gif"),
        embedding_mat_dir=Path("hyperbolicity/MNIST_0.999_WM_mat"),
    ),
}


def import_scipy_io():
    try:
        from scipy.io import loadmat, savemat
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: scipy. Install it in the environment used to run "
            "this script, for example: python -m pip install scipy"
        ) from exc
    return loadmat, savemat


def import_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as numpy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: numpy. Install it in the environment used to "
            "run this script, for example: python -m pip install numpy"
        ) from exc
    np = numpy
    return np


def import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it in the environment used "
            "to run this script, for example: python -m pip install matplotlib"
        ) from exc
    return plt, mpimg, FuncAnimation


def import_networkx():
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: networkx. Install it in the environment used "
            "to run this script, for example: python -m pip install networkx"
        ) from exc
    return nx


def parse_epochs(value: str) -> tuple[int, ...]:
    if ":" in value:
        parts = [int(part) if part else None for part in value.split(":")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError("Epoch range must look like start:stop[:step].")
        start = 0 if parts[0] is None else parts[0]
        stop = parts[1]
        step = 1 if len(parts) == 2 or parts[2] is None else parts[2]
        if stop is None:
            raise argparse.ArgumentTypeError("Epoch range needs a stop value.")
        return tuple(range(start, stop, step))
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_adjacencies(config: ExperimentConfig, epoch: int) -> list[np.ndarray]:
    loadmat, _ = import_scipy_io()
    adjs: list[np.ndarray] = []
    for layer in LAYER_NAMES:
        path = config.adjacency_root / f"adj_{layer}" / f"{epoch}.mat"
        require_file(path)
        mat = loadmat(path)
        if "adjacency_matrix" not in mat:
            raise KeyError(f"Missing 'adjacency_matrix' in {path}")
        adjs.append(np.asarray(mat["adjacency_matrix"]))
    validate_adjacency_shapes(adjs)
    return adjs


def validate_adjacency_shapes(adjs: Sequence[np.ndarray]) -> None:
    if len(adjs) != 3:
        raise ValueError(f"Expected 3 adjacency matrices, got {len(adjs)}.")
    if adjs[0].shape[1] != adjs[1].shape[0]:
        raise ValueError(f"Layer shape mismatch: {adjs[0].shape} -> {adjs[1].shape}.")
    if adjs[1].shape[1] != adjs[2].shape[0]:
        raise ValueError(f"Layer shape mismatch: {adjs[1].shape} -> {adjs[2].shape}.")


def active_nodes_for_path(adjs: Sequence[np.ndarray]) -> list[np.ndarray]:
    active = [np.flatnonzero(np.asarray(adjs[0].sum(axis=1)).reshape(-1))]
    for left, right in zip(adjs[:-1], adjs[1:]):
        incoming = np.asarray(left.sum(axis=0)).reshape(-1)
        outgoing = np.asarray(right.sum(axis=1)).reshape(-1)
        active.append(np.flatnonzero((incoming != 0) & (outgoing != 0)))
    active.append(np.flatnonzero(np.asarray(adjs[-1].sum(axis=0)).reshape(-1)))
    return active


def prune_active_path(
    adjs: Sequence[np.ndarray], iterations: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pruned = [np.asarray(adj).copy() for adj in adjs]
    active: list[np.ndarray] = []
    for _ in range(iterations):
        active = active_nodes_for_path(pruned)
        pruned = [adj[active[i]][:, active[i + 1]] for i, adj in enumerate(pruned)]
    return active, pruned


def graph_from_biadjacency(adjs: Sequence[np.ndarray], active_only: bool = False):
    nx = import_networkx()
    n0, n1 = adjs[0].shape
    _, n2 = adjs[1].shape
    _, n3 = adjs[2].shape

    if active_only:
        u_ids = np.flatnonzero(np.asarray(adjs[0].sum(axis=1)).reshape(-1))
        v_ids = np.union1d(
            np.flatnonzero(np.asarray(adjs[0].sum(axis=0)).reshape(-1)),
            np.flatnonzero(np.asarray(adjs[1].sum(axis=1)).reshape(-1)),
        )
        w_ids = np.union1d(
            np.flatnonzero(np.asarray(adjs[1].sum(axis=0)).reshape(-1)),
            np.flatnonzero(np.asarray(adjs[2].sum(axis=1)).reshape(-1)),
        )
        x_ids = np.flatnonzero(np.asarray(adjs[2].sum(axis=0)).reshape(-1))
    else:
        u_ids = np.arange(n0)
        v_ids = np.arange(n1)
        w_ids = np.arange(n2)
        x_ids = np.arange(n3)

    node_sets = (u_ids, v_ids, w_ids, x_ids)
    prefixes = ("u", "v", "w", "x")
    layer_sizes = (n0, n1, n2, n3)
    x_positions = (0, 10, 20, 30)
    max_size = max(layer_sizes)

    graph = nx.Graph()
    pos: dict[str, tuple[float, float]] = {}
    maps: list[dict[int, str]] = []
    for ids, prefix, layer_size, x_position in zip(
        node_sets, prefixes, layer_sizes, x_positions
    ):
        layer_map = {int(original): f"{prefix}{idx}" for idx, original in enumerate(ids)}
        maps.append(layer_map)
        for original, node in layer_map.items():
            graph.add_node(node)
            pos[node] = (x_position, original + (max_size - layer_size) / 2)

    for layer_idx, adj in enumerate(adjs):
        left_map = maps[layer_idx]
        right_map = maps[layer_idx + 1]
        for i, j in zip(*np.nonzero(adj)):
            left = left_map.get(int(i))
            right = right_map.get(int(j))
            if left is not None and right is not None:
                graph.add_edge(left, right)

    return graph, pos, node_sets


def active_layer_colors(node_sets: Sequence[np.ndarray]) -> list[str]:
    colors: list[str] = []
    for color, ids in zip(LAYER_HEX, node_sets):
        colors.extend([color] * len(ids))
    return colors


def pruned_layer_sizes(adjs: Sequence[np.ndarray]) -> tuple[int, int, int, int]:
    return (adjs[0].shape[0], adjs[0].shape[1], adjs[1].shape[1], adjs[2].shape[1])


def draw_plain_network_images(
    config: ExperimentConfig, epochs: Iterable[int], prune_iterations: int
) -> None:
    plt, _, _ = import_plotting()
    ensure_dir(config.plain_image_dir)

    for epoch in epochs:
        adjs = load_adjacencies(config, epoch)
        graph, pos, _ = graph_from_biadjacency(adjs, active_only=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axis_off()
        nx_kwargs = dict(width=0.005, node_size=5, node_shape="s", alpha=0.5, ax=ax)
        import_networkx().draw(graph, pos, node_color="grey", **nx_kwargs)

        active_graph, active_pos, node_sets = graph_from_biadjacency(adjs, active_only=True)
        active_colors = active_layer_colors(node_sets)
        import_networkx().draw(
            active_graph,
            active_pos,
            width=0.01,
            node_color=active_colors,
            node_size=5,
            node_shape="s",
            alpha=0.8,
            ax=ax,
        )

        _, pruned = prune_active_path(adjs, iterations=prune_iterations)
        print(epoch, "pruned layer sizes:", pruned_layer_sizes(pruned))

        out_path = config.plain_image_dir / f"{epoch}_plain.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_monopartite_adjacency(adjs: Sequence[np.ndarray]) -> np.ndarray:
    layer_sizes = pruned_layer_sizes(adjs)
    n0, n1, n2, n3 = layer_sizes
    total = sum(layer_sizes)
    adjacency = np.zeros((total, total), dtype=float)
    offsets = np.cumsum((0,) + layer_sizes)
    for idx, adj in enumerate(adjs):
        row_start = offsets[idx]
        row_end = offsets[idx + 1]
        col_start = offsets[idx + 1]
        col_end = offsets[idx + 2]
        adjacency[row_start:row_end, col_start:col_end] = adj
        adjacency[col_start:col_end, row_start:row_end] = adj.T
    adjacency[adjacency != 0] = 1.0
    return adjacency


def layer_labels(layer_sizes: Sequence[int]) -> np.ndarray:
    labels = []
    for idx, size in enumerate(layer_sizes, start=1):
        labels.extend([idx] * size)
    return np.asarray(labels, dtype=float).reshape(-1, 1)


def layer_color_matrix(labels: np.ndarray) -> np.ndarray:
    np = import_numpy()
    flat = labels.reshape(-1).astype(int)
    layer_rgb = np.asarray(LAYER_RGB_VALUES, dtype=float)
    return layer_rgb[flat - 1]


def connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
    n = adjacency.shape[0]
    neighbors = [np.flatnonzero(adjacency[i] != 0).astype(int) for i in range(n)]
    unseen = set(range(n))
    components: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp = [start]
        while stack:
            node = stack.pop()
            for nbr in neighbors[node]:
                nbr_int = int(nbr)
                if nbr_int in unseen:
                    unseen.remove(nbr_int)
                    stack.append(nbr_int)
                    comp.append(nbr_int)
        components.append(np.asarray(comp, dtype=int))
    return sorted(components, key=len, reverse=True)


def export_embedding_mat_files(
    config: ExperimentConfig,
    epochs: Iterable[int],
    prune_iterations: int,
    largest_component: bool,
) -> None:
    _, savemat = import_scipy_io()
    ensure_dir(config.embedding_mat_dir)

    for epoch in epochs:
        adjs = load_adjacencies(config, epoch)
        _, pruned = prune_active_path(adjs, iterations=prune_iterations)
        adjacency = build_monopartite_adjacency(pruned)
        labels = layer_labels(pruned_layer_sizes(pruned))
        components = connected_components(adjacency)
        if len(components) > 1:
            sizes = [len(comp) for comp in components]
            print(f"Warning: epoch {epoch} has {len(components)} components: {sizes}")
            if largest_component:
                keep = np.sort(components[0])
                adjacency = adjacency[np.ix_(keep, keep)]
                labels = labels[keep]

        colors = layer_color_matrix(labels)
        symbols = np.empty((adjacency.shape[0], 1), dtype=object)
        symbols[:, 0] = "o"

        out_path = config.embedding_mat_dir / f"{epoch}.mat"
        savemat(
            out_path,
            {
                "x_symetric": adjacency,
                "community": labels,
                "communitylayer": labels,
                "communitycolor": colors,
                "communitysymbol": symbols,
                "labels1": np.empty((0, 0)),
                "labels2": np.empty((0, 0)),
                "x_directional": np.empty((0, 0)),
            },
        )
        print(f"saved {out_path}")


def load_metric(path: Path, key: str = "top1") -> np.ndarray:
    loadmat, _ = import_scipy_io()
    require_file(path)
    mat = loadmat(path)
    if key not in mat:
        raise KeyError(f"Missing '{key}' in {path}")
    return np.asarray(mat[key]).reshape(-1)


def draw_summary_frames(
    config: ExperimentConfig, epochs: Sequence[int], prune_iterations: int
) -> None:
    plt, mpimg, _ = import_plotting()
    ensure_dir(config.frame_dir)

    fc = load_metric(config.fc_res_path)
    chts = load_metric(config.cht_res_path)
    anp_by_epoch: dict[int, float] = {}
    denominator = float(sum(config.layer_sizes))

    for epoch in epochs:
        adjs = load_adjacencies(config, epoch)
        _, pruned = prune_active_path(adjs, iterations=prune_iterations)
        sizes = pruned_layer_sizes(pruned)
        anp_by_epoch[epoch] = sum(sizes) / denominator

        plain_path = config.plain_image_dir / f"{epoch}_plain.png"
        require_file(plain_path)
        image = mpimg.imread(plain_path)

        fig = plt.figure(figsize=(15, 12), facecolor="white")
        grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
        ax_image = fig.add_subplot(grid[0, :])
        ax_acc = fig.add_subplot(grid[1, 0])
        ax_anp = fig.add_subplot(grid[1, 1])

        fig.suptitle(f"Epoch: {epoch}", fontsize=25)
        ax_image.imshow(image)
        ax_image.axis("off")
        ax_image.text(
            150,
            80,
            f"{sizes[0]}/{config.layer_sizes[0]}"
            f"                         {sizes[1]}/{config.layer_sizes[1]}"
            f"                         {sizes[2]}/{config.layer_sizes[2]}"
            f"                         {sizes[3]}/{config.layer_sizes[3]}",
            fontsize=12,
        )

        visible_epochs = [item for item in epochs if item <= epoch]
        ax_acc.plot(visible_epochs, fc[visible_epochs], color="black", label="FC")
        ax_acc.plot(
            visible_epochs,
            chts[visible_epochs],
            color="red",
            linestyle="dashed",
            label="CHTs",
        )
        ax_acc.set_xlabel("Epochs", fontsize=12)
        ax_acc.set_ylabel("Accuracy", fontsize=12)
        ax_acc.set_xticks([1, 25, 50, 75, 100])
        ax_acc.set_title("Accuracy", fontsize=15)
        ax_acc.legend()

        done_epochs = sorted(anp_by_epoch)
        ax_anp.plot(done_epochs, [1.0] * len(done_epochs), color="black", label="FC")
        ax_anp.plot(
            done_epochs,
            [anp_by_epoch[item] for item in done_epochs],
            color="red",
            linestyle="dashed",
            label="CHTs",
        )
        ax_anp.set_ylim(0.2, 1.1)
        ax_anp.legend()
        ax_anp.set_ylabel("ANP", fontsize=12)
        ax_anp.set_xlabel("Epochs", fontsize=12)
        ax_anp.set_title("ANP across the epochs", fontsize=15)
        ax_anp.set_xticks([1, 25, 50, 75, 100])

        out_path = config.frame_dir / f"full_{epoch}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


def save_gif(config: ExperimentConfig, epochs: Sequence[int], fps: int) -> None:
    plt, _, FuncAnimation = import_plotting()
    ensure_dir(config.gif_path.parent)
    image_files = [config.frame_dir / f"full_{epoch}.png" for epoch in epochs]
    for path in image_files:
        require_file(path)

    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_axis_off()
    image = plt.imread(image_files[0])
    im = ax.imshow(image)

    def update(frame: int):
        im.set_data(plt.imread(image_files[frame]))
        return [im]

    ani = FuncAnimation(fig, update, frames=len(image_files), blit=True)
    ani.save(config.gif_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"saved {config.gif_path}")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    root = Path(args.root).expanduser().resolve()
    config = CONFIGS[args.config].resolve(root)

    overrides = {}
    for field in (
        "adjacency_root",
        "fc_res_path",
        "cht_res_path",
        "plain_image_dir",
        "frame_dir",
        "gif_path",
        "embedding_mat_dir",
    ):
        value = getattr(args, field)
        if value is not None:
            path = Path(value).expanduser()
            overrides[field] = path if path.is_absolute() else root / path
    if args.layer_sizes is not None:
        sizes = tuple(int(item) for item in args.layer_sizes.split(","))
        if len(sizes) != 4:
            raise SystemExit("--layer-sizes must contain exactly four comma-separated sizes.")
        overrides["layer_sizes"] = sizes
    return replace(config, **overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plain", "frames", "gif", "export-mat", "all"),
        help="Pipeline step to run.",
    )
    parser.add_argument("--config", choices=sorted(CONFIGS), default="emnist")
    parser.add_argument("--root", default=".", help="Root for relative paths.")
    parser.add_argument("--epochs", type=parse_epochs, default=DEFAULT_EPOCHS)
    parser.add_argument("--prune-iterations", type=int, default=2)
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument("--largest-component", action="store_true")
    parser.add_argument("--layer-sizes")
    parser.add_argument("--adjacency-root")
    parser.add_argument("--fc-res-path")
    parser.add_argument("--cht-res-path")
    parser.add_argument("--plain-image-dir")
    parser.add_argument("--frame-dir")
    parser.add_argument("--gif-path")
    parser.add_argument("--embedding-mat-dir")
    args = parser.parse_args()

    import_numpy()
    config = build_config(args)
    epochs = tuple(args.epochs)

    if args.command in ("plain", "all"):
        draw_plain_network_images(config, epochs, args.prune_iterations)
    if args.command in ("export-mat", "all"):
        export_embedding_mat_files(
            config,
            epochs,
            args.prune_iterations,
            largest_component=args.largest_component,
        )
    if args.command in ("frames", "all"):
        draw_summary_frames(config, epochs, args.prune_iterations)
    if args.command in ("gif", "all"):
        save_gif(config, epochs, args.fps)


if __name__ == "__main__":
    main()
