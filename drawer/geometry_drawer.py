#!/usr/bin/env python
"""Numerical drawer for small formal geometry descriptions."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


EPS = 1e-9


@dataclass(frozen=True)
class Command:
    op: str
    args: Tuple[str, ...]
    line_no: int
    raw: str


POINT_ARITIES = {
    "cong": 4,
    "coll": 3,
    "cyclic": 4,
    "eqangle": (6, 8),
    "midpoint": 3,
    "parallel": 4,
    "perp": 4,
    "check_cyclic": 4,
    "check_eqangle": (6, 8),
}


def parse_text(text: str) -> List[Command]:
    commands: List[Command] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        raw = raw_line.split("#", 1)[0].strip()
        if not raw:
            continue
        parts = raw.split()
        op = parts[0].lower()
        args = tuple(parts[1:])

        if op in ("fix", "init"):
            if len(args) != 3:
                raise ValueError(f"line {line_no}: {op} expects: point x y")
            float(args[1])
            float(args[2])
        elif op in POINT_ARITIES:
            arity = POINT_ARITIES[op]
            ok = len(args) in arity if isinstance(arity, tuple) else len(args) == arity
            if not ok:
                raise ValueError(f"line {line_no}: {op} has wrong arity")
        else:
            known = ", ".join(sorted(["fix", "init"] + list(POINT_ARITIES)))
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
        else:
            names.update(cmd.args)
    return sorted(names)


def cross2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def dot2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (u * v).sum(dim=-1)


def norm(v: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(dot2(v, v) + EPS)


def distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return norm(p - q)


def angle_at(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    u = a - b
    v = c - b
    return torch.atan2(torch.abs(cross2(u, v)), dot2(u, v))


def oriented_angle_at(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    u = a - b
    v = c - b
    return torch.atan2(cross2(u, v), dot2(u, v))


def directed_line_angle(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    u = b - a
    v = d - c
    return torch.atan2(cross2(u, v), dot2(u, v))


def wrap_two_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def wrap_line_angle(x: torch.Tensor) -> torch.Tensor:
    # Oriented angles between straight lines are taken modulo pi.
    return 0.5 * torch.atan2(torch.sin(2.0 * x), torch.cos(2.0 * x))


def collinearity_loss(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    def sin_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cross2(x, y).pow(2) / (dot2(x, x) * dot2(y, y) + EPS)

    return (
        sin_sq(b - a, c - a)
        + sin_sq(a - b, c - b)
        + sin_sq(a - c, b - c)
    )


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
        return xy.to(dtype=torch.float64)

    def clamp_fixed(self, xy: torch.Tensor) -> None:
        with torch.no_grad():
            for name, value in self.fixed.items():
                xy[self.index[name]] = torch.tensor(value, dtype=xy.dtype, device=xy.device)

    def loss_terms(self, xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        terms: Dict[str, List[torch.Tensor]] = {
            "cong": [],
            "coll": [],
            "cyclic": [],
            "eqangle": [],
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
            elif cmd.op == "parallel":
                a, b, c, d = [self.p(xy, x) for x in args]
                u = b - a
                v = d - c
                terms["parallel"].append(cross2(u, v).pow(2) / (dot2(u, u) * dot2(v, v) + EPS))
            elif cmd.op == "perp":
                a, b, c, d = [self.p(xy, x) for x in args]
                u = b - a
                v = d - c
                terms["perp"].append(dot2(u, v).pow(2) / (dot2(u, u) * dot2(v, v) + EPS))

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

    def optimize(
        self,
        steps: int,
        lr: float,
        restarts: int,
        separation_weight: float,
        seed: int,
        init_noise: float,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        best_xy: Optional[torch.Tensor] = None
        best_terms: Optional[Dict[str, float]] = None
        best_score = float("inf")

        for restart in range(restarts):
            xy = self.initial_tensor(seed + 1009 * restart, init_noise).detach().requires_grad_(True)
            opt = torch.optim.Adam([xy], lr=lr)

            for step in range(steps):
                opt.zero_grad()
                terms = self.loss_terms(xy)
                constraint_loss = sum(
                    value for key, value in terms.items() if key != "separation"
                )
                loss = constraint_loss + separation_weight * terms["separation"]
                loss.backward()
                opt.step()
                self.clamp_fixed(xy)

                if verbose and (step + 1) % max(1, steps // 10) == 0:
                    print(
                        f"restart={restart + 1}/{restarts} step={step + 1}/{steps} "
                        f"loss={loss.item():.6g}"
                    )

            with torch.no_grad():
                terms = self.loss_terms(xy)
                constraint_loss = sum(
                    value for key, value in terms.items() if key != "separation"
                )
                score = (constraint_loss + separation_weight * terms["separation"]).item()
                if score < best_score:
                    best_score = score
                    best_xy = xy.detach().clone()
                    best_terms = {k: v.item() for k, v in terms.items()}

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
        elif cmd.op in ("parallel", "perp"):
            segment(args[0], args[1], "-", 0.5)
            segment(args[2], args[3], "-", 0.5)
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
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-noise", type=float, default=0.03)
    parser.add_argument("--separation-weight", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot", help="optional output png path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    commands = load_commands(args.input)
    problem = GeometryProblem(commands, device=args.device)
    xy, terms = problem.optimize(
        steps=args.steps,
        lr=args.lr,
        restarts=args.restarts,
        separation_weight=args.separation_weight,
        seed=args.seed,
        init_noise=args.init_noise,
        verbose=args.verbose,
    )
    if args.plot:
        plot_solution(args.plot, problem, problem.coordinates(xy), commands)
    print(format_report(problem, xy, terms, args.separation_weight))
    if args.plot:
        print("")
        print(f"plot {os.path.abspath(args.plot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
