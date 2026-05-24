#!/usr/bin/env python
"""Numerical drawer for small formal geometry descriptions."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from geometry_constraints import (
    EPS,
    angle_at,
    angles_equation_loss,
    collinearity_loss,
    convex_polygon_loss,
    cross2,
    directed_line_angle,
    distance,
    distances_equation_loss,
    dot2,
    eqratio_loss,
    excenter,
    incenter,
    line_intersection,
    norm,
    oriented_angle_at,
    orthocenter,
    parallel_loss,
    perpendicular_loss,
    point_match_loss,
    wrap_line_angle,
    wrap_two_pi,
)


@dataclass(frozen=True)
class Command:
    op: str
    args: Tuple[str, ...]
    line_no: int
    raw: str


@dataclass
class RestartProgress:
    restart: int
    step: int = 0
    loss: float = float("nan")
    constraint_loss: float = float("nan")
    separation: float = float("nan")
    best_loss: float = float("inf")
    status: str = "pending"


POINT_ARITIES = {
    "cong": 4,
    "coll": 3,
    "cyclic": 4,
    "eqangle": (6, 8),
    "eqratio": 8,
    "midpoint": 3,
    "para": 4,
    "parallel": 4,
    "perp": 4,
    "check_cyclic": 4,
    "check_eqangle": (6, 8),
}

VARIABLE_ARITY_COMMANDS = {
    "angles_equation",
    "convex_polygon",
    "dists_equation",
}

CONSTRUCTION_ARITIES = {
    "excenter": 3,
    "incenter": 3,
    "intersect": 4,
    "orthocenter": 3,
}


def parse_text(text: str) -> List[Command]:
    commands: List[Command] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        raw = raw_line.split("#", 1)[0].strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) >= 3 and parts[1] == "=":
            op = parts[2].lower()
            args = tuple([parts[0]] + parts[3:])
        else:
            op = parts[0].lower()
            args = tuple(parts[1:])

        if op in ("fix", "init"):
            if len(args) != 3:
                raise ValueError(f"line {line_no}: {op} expects: point x y")
            float(args[1])
            float(args[2])
        elif op in CONSTRUCTION_ARITIES:
            arity = CONSTRUCTION_ARITIES[op]
            if len(args) != arity + 1:
                raise ValueError(f"line {line_no}: {op} expects: target = {op} points...")
            if not (len(parts) >= 3 and parts[1] == "="):
                raise ValueError(f"line {line_no}: {op} must be written as: point = {op} ...")
        elif op in POINT_ARITIES:
            arity = POINT_ARITIES[op]
            ok = len(args) in arity if isinstance(arity, tuple) else len(args) == arity
            if not ok:
                raise ValueError(f"line {line_no}: {op} has wrong arity")
        elif op == "dists_equation":
            if len(args) < 4 or (len(args) - 1) % 3 != 0:
                raise ValueError(
                    f"line {line_no}: dists_equation expects: v1 a1 b1 ... vn an bn target"
                )
            for i in range(0, len(args) - 1, 3):
                float(args[i])
            float(args[-1])
        elif op == "angles_equation":
            if len(args) < 5 or (len(args) - 1) % 4 != 0:
                raise ValueError(
                    f"line {line_no}: angles_equation expects: v1 a1 b1 c1 ... vn an bn cn target"
                )
            for i in range(0, len(args) - 1, 4):
                float(args[i])
            float(args[-1])
        elif op == "convex_polygon":
            if len(args) < 3:
                raise ValueError(f"line {line_no}: convex_polygon expects at least 3 points")
        else:
            known = ", ".join(
                sorted(
                    ["fix", "init"]
                    + list(POINT_ARITIES)
                    + list(VARIABLE_ARITY_COMMANDS)
                    + list(CONSTRUCTION_ARITIES)
                )
            )
            raise ValueError(f"line {line_no}: unknown command {op!r}; known: {known}")

        commands.append(Command(op, args, line_no, raw))
    return commands


def load_commands(path: str) -> List[Command]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_text(f.read())


def point_names(commands: Sequence[Command]) -> List[str]:
    names = set()
    for cmd in commands:
        if cmd.op in ("fix", "init"):
            names.add(cmd.args[0])
        elif cmd.op == "dists_equation":
            for i in range(0, len(cmd.args) - 1, 3):
                names.add(cmd.args[i + 1])
                names.add(cmd.args[i + 2])
        elif cmd.op == "angles_equation":
            for i in range(0, len(cmd.args) - 1, 4):
                names.update(cmd.args[i + 1 : i + 4])
        else:
            names.update(cmd.args)
    return sorted(names)


def circumcircle(points: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    (x1, y1), (x2, y2), (x3, y3) = points[:3]
    det = 2.0 * (
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )
    if abs(det) < 1e-10:
        return None
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / det
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / det
    r = math.hypot(x1 - ux, y1 - uy)
    return ux, uy, r


class GeometryProblem:
    def __init__(self, commands: Sequence[Command], device: str = "cpu") -> None:
        self.commands = list(commands)
        self.names = point_names(commands)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.device = torch.device(device)
        self.fixed = self._fixed_points()
        self.inits = self._init_points()

    def _fixed_points(self) -> Dict[str, Tuple[float, float]]:
        fixed: Dict[str, Tuple[float, float]] = {}
        for cmd in self.commands:
            if cmd.op == "fix":
                fixed[cmd.args[0]] = (float(cmd.args[1]), float(cmd.args[2]))
        return fixed

    def _init_points(self) -> Dict[str, Tuple[float, float]]:
        inits: Dict[str, Tuple[float, float]] = {}
        for cmd in self.commands:
            if cmd.op == "init":
                inits[cmd.args[0]] = (float(cmd.args[1]), float(cmd.args[2]))
        return inits

    def p(self, xy: torch.Tensor, name: str) -> torch.Tensor:
        return xy[self.index[name]]

    def constructed_point(self, xy: torch.Tensor, cmd: Command) -> torch.Tensor:
        args = cmd.args
        if cmd.op == "intersect":
            _, a, b, c, d = [self.p(xy, name) for name in args]
            return line_intersection(a, b, c, d)
        if cmd.op == "orthocenter":
            _, a, b, c = [self.p(xy, name) for name in args]
            return orthocenter(a, b, c)
        if cmd.op == "incenter":
            _, a, b, c = [self.p(xy, name) for name in args]
            return incenter(a, b, c)
        if cmd.op == "excenter":
            _, a, b, c = [self.p(xy, name) for name in args]
            return excenter(a, b, c)
        raise ValueError(f"line {cmd.line_no}: {cmd.op} is not a construction")

    def initialize_constructed_points(self, xy: torch.Tensor) -> None:
        for cmd in self.commands:
            if cmd.op not in CONSTRUCTION_ARITIES:
                continue
            target = cmd.args[0]
            if target in self.fixed:
                continue
            xy[self.index[target]] = self.constructed_point(xy, cmd)

    def initial_tensor(self, seed: int, noise: float) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        xy = torch.randn((len(self.names), 2), generator=gen, device=self.device) * 0.7
        for name, value in self.inits.items():
            i = self.index[name]
            base = torch.tensor(value, dtype=torch.float64, device=self.device)
            xy[i] = base + noise * torch.randn((2,), generator=gen, device=self.device)
        for name, value in self.fixed.items():
            xy[self.index[name]] = torch.tensor(value, dtype=torch.float64, device=self.device)
        xy = xy.to(dtype=torch.float64)
        self.initialize_constructed_points(xy)
        self.clamp_fixed(xy)
        return xy

    def clamp_fixed(self, xy: torch.Tensor) -> None:
        with torch.no_grad():
            for name, value in self.fixed.items():
                xy[self.index[name]] = torch.tensor(value, dtype=xy.dtype, device=xy.device)

    def loss_terms(self, xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        terms: Dict[str, List[torch.Tensor]] = {
            "angles_equation": [],
            "cong": [],
            "coll": [],
            "construction": [],
            "convex_polygon": [],
            "cyclic": [],
            "dists_equation": [],
            "eqangle": [],
            "eqratio": [],
            "midpoint": [],
            "parallel": [],
            "perp": [],
        }

        for cmd in self.commands:
            args = cmd.args
            if cmd.op in ("fix", "init", "check_cyclic", "check_eqangle"):
                continue

            if cmd.op == "cong":
                a, b, c, d = [self.p(xy, x) for x in args]
                terms["cong"].append((distance(a, b) - distance(c, d)).pow(2))
            elif cmd.op == "coll":
                a, b, c = [self.p(xy, x) for x in args]
                terms["coll"].append(collinearity_loss(a, b, c))
            elif cmd.op == "cyclic":
                a, b, c, d = [self.p(xy, x) for x in args]
                diff = wrap_two_pi(oriented_angle_at(a, b, d) - oriented_angle_at(a, c, d))
                terms["cyclic"].append(diff.pow(2))
            elif cmd.op == "eqangle" and len(args) == 6:
                a, b, c, d, e, f = [self.p(xy, x) for x in args]
                terms["eqangle"].append((angle_at(a, b, c) - angle_at(d, e, f)).pow(2))
            elif cmd.op == "eqangle" and len(args) == 8:
                a, b, c, d, e, f, g, h = [self.p(xy, x) for x in args]
                diff = wrap_line_angle(
                    directed_line_angle(a, b, c, d) - directed_line_angle(e, f, g, h)
                )
                terms["eqangle"].append(diff.pow(2))
            elif cmd.op == "midpoint":
                m, a, b = [self.p(xy, x) for x in args]
                terms["midpoint"].append(((m - 0.5 * (a + b)).pow(2)).sum())
            elif cmd.op in ("parallel", "para"):
                a, b, c, d = [self.p(xy, x) for x in args]
                terms["parallel"].append(parallel_loss(a, b, c, d))
            elif cmd.op == "perp":
                a, b, c, d = [self.p(xy, x) for x in args]
                terms["perp"].append(perpendicular_loss(a, b, c, d))
            elif cmd.op == "eqratio":
                points = [self.p(xy, x) for x in args]
                terms["eqratio"].append(eqratio_loss(points))
            elif cmd.op == "dists_equation":
                coefficients = [float(args[i]) for i in range(0, len(args) - 1, 3)]
                segments = [
                    (self.p(xy, args[i + 1]), self.p(xy, args[i + 2]))
                    for i in range(0, len(args) - 1, 3)
                ]
                terms["dists_equation"].append(
                    distances_equation_loss(coefficients, segments, float(args[-1]))
                )
            elif cmd.op == "angles_equation":
                coefficients = [float(args[i]) for i in range(0, len(args) - 1, 4)]
                angles = [
                    (self.p(xy, args[i + 1]), self.p(xy, args[i + 2]), self.p(xy, args[i + 3]))
                    for i in range(0, len(args) - 1, 4)
                ]
                terms["angles_equation"].append(
                    angles_equation_loss(coefficients, angles, float(args[-1]))
                )
            elif cmd.op == "convex_polygon":
                terms["convex_polygon"].append(convex_polygon_loss([self.p(xy, x) for x in args]))
            elif cmd.op in CONSTRUCTION_ARITIES:
                terms["construction"].append(
                    point_match_loss(self.p(xy, args[0]), self.constructed_point(xy, cmd))
                )

        out: Dict[str, torch.Tensor] = {}
        zero = torch.tensor(0.0, dtype=xy.dtype, device=xy.device)
        for key, values in terms.items():
            out[key] = torch.stack(values).sum() if values else zero
        out["separation"] = self.separation_penalty(xy)
        return out

    def separation_penalty(self, xy: torch.Tensor) -> torch.Tensor:
        parts = []
        for i in range(len(self.names)):
            for j in range(i + 1, len(self.names)):
                dist2 = ((xy[i] - xy[j]).pow(2)).sum()
                parts.append(torch.exp(-40.0 * dist2))
        if not parts:
            return torch.tensor(0.0, dtype=xy.dtype, device=xy.device)
        return torch.stack(parts).sum()

    def total_loss(
        self,
        terms: Dict[str, torch.Tensor],
        separation_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        constraint_loss = sum(value for key, value in terms.items() if key != "separation")
        loss = constraint_loss + separation_weight * terms["separation"]
        return loss, constraint_loss

    def _optimize_restart(
        self,
        restart: int,
        steps: int,
        lr: float,
        separation_weight: float,
        seed: int,
        init_noise: float,
        success_loss: Optional[float],
        stop_event: threading.Event,
        progress: Dict[int, RestartProgress],
        progress_lock: threading.Lock,
    ) -> Optional[Tuple[float, torch.Tensor, Dict[str, float]]]:
        with progress_lock:
            progress[restart].status = "running"

        xy = self.initial_tensor(seed + 1009 * restart, init_noise).detach().requires_grad_(True)
        opt = torch.optim.Adam([xy], lr=lr)
        best_score = float("inf")
        best_xy: Optional[torch.Tensor] = None
        best_terms: Optional[Dict[str, float]] = None

        for step in range(steps):
            if stop_event.is_set():
                with progress_lock:
                    if progress[restart].status != "success":
                        progress[restart].status = "stopped"
                break

            opt.zero_grad()
            terms = self.loss_terms(xy)
            loss, constraint_loss = self.total_loss(terms, separation_weight)
            loss.backward()
            opt.step()
            self.clamp_fixed(xy)

            score = loss.item()
            constraint_score = constraint_loss.item()
            separation_score = terms["separation"].item()
            if score < best_score:
                best_score = score
                best_xy = xy.detach().clone()
                best_terms = {k: v.item() for k, v in terms.items()}

            with progress_lock:
                row = progress[restart]
                row.step = step + 1
                row.loss = score
                row.constraint_loss = constraint_score
                row.separation = separation_score
                row.best_loss = best_score

            if success_loss is not None and constraint_score <= success_loss:
                stop_event.set()
                with progress_lock:
                    progress[restart].status = "success"
                break
        else:
            with progress_lock:
                progress[restart].status = "done"

        if best_xy is None or best_terms is None:
            with torch.no_grad():
                terms = self.loss_terms(xy)
                loss, _ = self.total_loss(terms, separation_weight)
                best_score = loss.item()
                best_xy = xy.detach().clone()
                best_terms = {k: v.item() for k, v in terms.items()}

        return best_score, best_xy, best_terms

    def _progress_table(self, progress: Dict[int, RestartProgress], workers: int):
        from rich.table import Table

        table = Table(title=f"Geometry optimization ({workers} workers)")
        table.add_column("Restart", justify="right")
        table.add_column("Status")
        table.add_column("Step", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Constraint", justify="right")
        table.add_column("Separation", justify="right")
        table.add_column("Best", justify="right")

        def fmt(value: float) -> str:
            if math.isnan(value):
                return "-"
            if math.isinf(value):
                return "-"
            return f"{value:.4g}"

        for restart in sorted(progress):
            row = progress[restart]
            table.add_row(
                str(row.restart + 1),
                row.status,
                str(row.step),
                fmt(row.loss),
                fmt(row.constraint_loss),
                fmt(row.separation),
                fmt(row.best_loss),
            )
        return table

    def optimize(
        self,
        steps: int,
        lr: float,
        restarts: int,
        separation_weight: float,
        seed: int,
        init_noise: float,
        verbose: bool = False,
        workers: int = 1,
        success_loss: Optional[float] = None,
        live_progress: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        best_xy: Optional[torch.Tensor] = None
        best_terms: Optional[Dict[str, float]] = None
        best_score = float("inf")

        if restarts < 1:
            raise ValueError("restarts must be at least 1")
        if workers < 1:
            raise ValueError("workers must be at least 1")

        worker_count = min(workers, restarts)
        progress = {restart: RestartProgress(restart) for restart in range(restarts)}
        progress_lock = threading.Lock()
        stop_event = threading.Event()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._optimize_restart,
                    restart,
                    steps,
                    lr,
                    separation_weight,
                    seed,
                    init_noise,
                    success_loss,
                    stop_event,
                    progress,
                    progress_lock,
                ): restart
                for restart in range(restarts)
            }

            pending = set(futures)
            if live_progress:
                from rich.live import Live

                with Live(self._progress_table(progress, worker_count), refresh_per_second=6) as live:
                    while pending:
                        done, pending = concurrent.futures.wait(
                            pending,
                            timeout=0.15,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            score, xy, terms = future.result()
                            if score < best_score:
                                best_score = score
                                best_xy = xy
                                best_terms = terms
                        with progress_lock:
                            live.update(self._progress_table(progress, worker_count))
                    live.update(self._progress_table(progress, worker_count))
            else:
                for future in concurrent.futures.as_completed(futures):
                    score, xy, terms = future.result()
                    if score < best_score:
                        best_score = score
                        best_xy = xy
                        best_terms = terms
                    if verbose:
                        restart = futures[future]
                        print(
                            f"restart={restart + 1}/{restarts} status={progress[restart].status} "
                            f"best_loss={score:.6g}"
                        )

        assert best_xy is not None and best_terms is not None
        return best_xy.cpu(), best_terms

    def coordinates(self, xy: torch.Tensor) -> Dict[str, Tuple[float, float]]:
        return {
            name: (float(xy[self.index[name], 0]), float(xy[self.index[name], 1]))
            for name in self.names
        }

    def checks(self, xy: torch.Tensor) -> List[Tuple[Command, float]]:
        out: List[Tuple[Command, float]] = []
        for cmd in self.commands:
            if cmd.op == "check_cyclic":
                a, b, c, d = [self.p(xy, x) for x in cmd.args]
                residual = wrap_two_pi(oriented_angle_at(a, b, d) - oriented_angle_at(a, c, d)).abs().item()
                out.append((cmd, residual))
                continue
            if cmd.op != "check_eqangle":
                continue
            args = cmd.args
            if len(args) == 6:
                a, b, c, d, e, f = [self.p(xy, x) for x in args]
                residual = (angle_at(a, b, c) - angle_at(d, e, f)).abs().item()
            else:
                a, b, c, d, e, f, g, h = [self.p(xy, x) for x in args]
                residual = wrap_line_angle(
                    directed_line_angle(a, b, c, d) - directed_line_angle(e, f, g, h)
                ).abs().item()
            out.append((cmd, residual))
        return out

    def normalized_commands(self) -> List[str]:
        lines = []
        for cmd in self.commands:
            if cmd.op == "eqangle" and len(cmd.args) == 6:
                a, b, c, d, e, f = cmd.args
                lines.append(f"eqangle {a} {b} {b} {c} {d} {e} {e} {f}")
            elif cmd.op == "check_eqangle" and len(cmd.args) == 6:
                a, b, c, d, e, f = cmd.args
                lines.append(f"check_eqangle {a} {b} {b} {c} {d} {e} {e} {f}")
            elif cmd.op in CONSTRUCTION_ARITIES:
                lines.append(f"{cmd.args[0]} = {cmd.op} {' '.join(cmd.args[1:])}")
            else:
                lines.append(f"{cmd.op} {' '.join(cmd.args)}")
        return lines


def plot_solution(
    path: str,
    problem: GeometryProblem,
    coords: Dict[str, Tuple[float, float]],
    commands: Sequence[Command],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))

    def pt(name: str) -> Tuple[float, float]:
        return coords[name]

    drawn_segments = set()

    def segment(a: str, b: str, style: str = "-", alpha: float = 0.55) -> None:
        key = tuple(sorted((a, b)))
        if key in drawn_segments:
            return
        drawn_segments.add(key)
        x1, y1 = pt(a)
        x2, y2 = pt(b)
        ax.plot([x1, x2], [y1, y2], style, color="#555555", alpha=alpha, linewidth=1.2)

    for cmd in commands:
        args = cmd.args
        if cmd.op == "cong":
            segment(args[0], args[1], "--", 0.8)
            segment(args[2], args[3], "--", 0.8)
        elif cmd.op == "coll":
            segment(args[0], args[1])
            segment(args[1], args[2])
        elif cmd.op == "cyclic":
            pts = [pt(name) for name in args[:3]]
            circle = circumcircle(pts)
            if circle:
                ux, uy, r = circle
                ax.add_patch(Circle((ux, uy), r, fill=False, linestyle=":", color="#777777"))
            for i in range(len(args)):
                segment(args[i], args[(i + 1) % len(args)], ":", 0.35)
        elif cmd.op == "midpoint":
            segment(args[1], args[2], "-", 0.25)
        elif cmd.op in ("parallel", "para", "perp", "eqratio"):
            segment(args[0], args[1], "-", 0.5)
            segment(args[2], args[3], "-", 0.5)
            if cmd.op == "eqratio":
                segment(args[4], args[5], "-", 0.5)
                segment(args[6], args[7], "-", 0.5)
        elif cmd.op == "dists_equation":
            for i in range(0, len(args) - 1, 3):
                segment(args[i + 1], args[i + 2], "-", 0.45)
        elif cmd.op == "angles_equation":
            for i in range(0, len(args) - 1, 4):
                segment(args[i + 2], args[i + 1], "-", 0.25)
                segment(args[i + 2], args[i + 3], "-", 0.25)
        elif cmd.op == "convex_polygon":
            for i in range(len(args)):
                segment(args[i], args[(i + 1) % len(args)], "-", 0.65)
        elif cmd.op == "intersect":
            segment(args[1], args[2], "-", 0.55)
            segment(args[3], args[4], "-", 0.55)
        elif cmd.op in ("orthocenter", "incenter", "excenter"):
            segment(args[1], args[2], "-", 0.35)
            segment(args[2], args[3], "-", 0.35)
            segment(args[3], args[1], "-", 0.35)
            segment(args[0], args[1], ":", 0.25)
            segment(args[0], args[2], ":", 0.25)
            segment(args[0], args[3], ":", 0.25)
        elif cmd.op == "eqangle":
            if len(args) == 6:
                segment(args[1], args[0], "-", 0.25)
                segment(args[1], args[2], "-", 0.25)
                segment(args[4], args[3], "-", 0.25)
                segment(args[4], args[5], "-", 0.25)
            else:
                segment(args[0], args[1], "-", 0.25)
                segment(args[2], args[3], "-", 0.25)
                segment(args[4], args[5], "-", 0.25)
                segment(args[6], args[7], "-", 0.25)

    xs = [xy[0] for xy in coords.values()]
    ys = [xy[1] for xy in coords.values()]
    ax.scatter(xs, ys, s=32, color="#111111", zorder=3)
    for name, (x, y) in coords.items():
        ax.annotate(name, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=11)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#eeeeee")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_report(
    problem: GeometryProblem,
    xy: torch.Tensor,
    terms: Dict[str, float],
    separation_weight: float,
) -> str:
    coords = problem.coordinates(xy)
    constraint_loss = sum(value for key, value in terms.items() if key != "separation")
    total_loss = constraint_loss + separation_weight * terms["separation"]

    lines = []
    lines.append(f"total_loss {total_loss:.12g}")
    lines.append(f"constraint_loss {constraint_loss:.12g}")
    lines.append(f"separation_penalty {terms['separation']:.12g}")
    lines.append("")
    lines.append("coordinates")
    for name, (x, y) in coords.items():
        lines.append(f"{name} {x:.10f} {y:.10f}")
    lines.append("")
    lines.append("loss_terms")
    for key in sorted(terms):
        lines.append(f"{key} {terms[key]:.12g}")
    checks = problem.checks(xy)
    if checks:
        lines.append("")
        lines.append("checks")
        for cmd, residual in checks:
            lines.append(f"line {cmd.line_no}: {cmd.raw} residual_rad {residual:.12g}")
    lines.append("")
    lines.append("normalized_commands")
    lines.extend(problem.normalized_commands())
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="path to a geometry description")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-noise", type=float, default=0.03)
    parser.add_argument("--separation-weight", type=float, default=0.001)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="parallel optimization workers for independent restarts",
    )
    parser.add_argument(
        "--success-loss",
        type=float,
        default=1e-8,
        help="stop all workers once constraint loss reaches this value; negative disables early stop",
    )
    parser.add_argument("--no-live", action="store_true", help="disable rich live progress table")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot", help="optional output png path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    commands = load_commands(args.input)
    problem = GeometryProblem(commands, device=args.device)
    live_progress = args.threads > 1 and not args.no_live
    xy, terms = problem.optimize(
        steps=args.steps,
        lr=args.lr,
        restarts=args.restarts,
        separation_weight=args.separation_weight,
        seed=args.seed,
        init_noise=args.init_noise,
        verbose=args.verbose,
        workers=args.threads,
        success_loss=None if args.success_loss < 0 else args.success_loss,
        live_progress=live_progress,
    )
    if args.plot:
        plot_solution(args.plot, problem, problem.coordinates(xy), commands)
    if live_progress:
        print("")
    print(format_report(problem, xy, terms, args.separation_weight))
    if args.plot:
        print("")
        print(f"plot {os.path.abspath(args.plot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
