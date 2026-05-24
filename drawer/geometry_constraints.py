"""Differentiable geometry loss primitives for geometry_drawer."""

from __future__ import annotations

from typing import Sequence

import torch


EPS = 1e-9


def cross2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def dot2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (u * v).sum(dim=-1)


def norm(v: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(dot2(v, v) + EPS)


def distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return norm(p - q)


def rotate90(v: torch.Tensor) -> torch.Tensor:
    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


def signed_safe_denominator(value: torch.Tensor) -> torch.Tensor:
    sign = torch.where(value.detach() >= 0, value.new_tensor(1.0), value.new_tensor(-1.0))
    return value + sign * EPS


def line_intersection(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    u = b - a
    v = d - c
    t = cross2(c - a, v) / signed_safe_denominator(cross2(u, v))
    return a + t * u


def orthocenter(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    altitude_a_end = a + rotate90(c - b)
    altitude_b_end = b + rotate90(c - a)
    return line_intersection(a, altitude_a_end, b, altitude_b_end)


def incenter(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    side_a = distance(b, c)
    side_b = distance(c, a)
    side_c = distance(a, b)
    return (side_a * a + side_b * b + side_c * c) / (side_a + side_b + side_c + EPS)


def excenter(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    side_a = distance(b, c)
    side_b = distance(c, a)
    side_c = distance(a, b)
    return (-side_a * a + side_b * b + side_c * c) / (-side_a + side_b + side_c + EPS)


def point_match_loss(target: torch.Tensor, constructed: torch.Tensor) -> torch.Tensor:
    return (target - constructed).pow(2).sum()


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


def parallel_loss(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    u = b - a
    v = d - c
    return cross2(u, v).pow(2) / (dot2(u, u) * dot2(v, v) + EPS)


def perpendicular_loss(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    u = b - a
    v = d - c
    return dot2(u, v).pow(2) / (dot2(u, u) * dot2(v, v) + EPS)


def eqratio_loss(points: Sequence[torch.Tensor]) -> torch.Tensor:
    a, b, c, d, e, f, g, h = points
    return (distance(a, b) / (distance(c, d) + EPS) - distance(e, f) / (distance(g, h) + EPS)).pow(2)


def distances_equation_loss(
    coefficients: Sequence[float],
    segments: Sequence[tuple[torch.Tensor, torch.Tensor]],
    target: float,
) -> torch.Tensor:
    residual = torch.as_tensor(target, dtype=segments[0][0].dtype, device=segments[0][0].device)
    residual = -residual
    for coefficient, (a, b) in zip(coefficients, segments):
        residual = residual + coefficient * distance(a, b)
    return residual.pow(2)


def angles_equation_loss(
    coefficients: Sequence[float],
    angles: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    target: float,
) -> torch.Tensor:
    residual = torch.as_tensor(target, dtype=angles[0][0].dtype, device=angles[0][0].device)
    residual = -residual
    for coefficient, (a, b, c) in zip(coefficients, angles):
        residual = residual + coefficient * angle_at(a, b, c)
    return residual.pow(2)


def convex_polygon_loss(points: Sequence[torch.Tensor]) -> torch.Tensor:
    signed_area = points[0].new_tensor(0.0)
    turns = []
    n = len(points)
    for i in range(n):
        p = points[i]
        q = points[(i + 1) % n]
        r = points[(i + 2) % n]
        signed_area = signed_area + cross2(p, q)
        u = q - p
        v = r - q
        denom = norm(u) * norm(v) + EPS
        turns.append(cross2(u, v) / denom)

    orientation = torch.where(
        signed_area.detach() >= 0,
        signed_area.new_tensor(1.0),
        signed_area.new_tensor(-1.0),
    )
    return torch.stack([torch.relu(-orientation * turn).pow(2) for turn in turns]).sum()
