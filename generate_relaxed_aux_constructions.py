from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


INITIAL_PROBLEMS = [
    ("a b = segment a b", ["a", "b"]),
    ("a b c = triangle a b c", ["a", "b", "c"]),
    ("a b c = acute_triangle a b c", ["a", "b", "c"]),
    ("a b c = r_triangle a b c", ["a", "b", "c"]),
    ("a b c d = quadrangle a b c d", ["a", "b", "c", "d"]),
]


# name, number of output points, number of already-existing input points
CONSTRUCTIONS = [
    ("free", 1, 0),
    ("segment", 2, 0),
    ("triangle", 3, 0),
    ("r_triangle", 3, 0),
    ("acute_triangle", 3, 0),
    ("quadrangle", 4, 0),
    ("rectangle", 4, 0),
    ("on_line", 1, 2),
    ("on_pline", 1, 3),
    ("on_pline0", 1, 3),
    ("on_tline", 1, 3),
    ("on_aline", 1, 5),
    ("on_aline0", 1, 7),
    ("on_bline", 1, 2),
    ("on_circle", 1, 2),
    ("on_circum", 1, 3),
    ("on_dia", 1, 2),
    ("intersection_ll", 1, 4),
    ("intersection_lp", 1, 5),
    ("intersection_pp", 1, 6),
    ("intersection_lt", 1, 5),
    ("intersection_tt", 1, 6),
    ("intersection_lc", 1, 3),
    ("intersection_cc", 1, 3),
    ("tangent", 2, 3),
    ("lc_tangent", 1, 2),
    ("angle_bisector", 1, 3),
    ("angle_mirror", 1, 3),
    ("eqdistance", 1, 3),
    ("foot", 1, 3),
    ("incenter", 1, 3),
    ("incenter2", 4, 3),
    ("excenter", 1, 3),
    ("excenter2", 4, 3),
    ("midpoint", 1, 2),
    ("mirror", 1, 2),
    ("orthocenter", 1, 3),
    ("reflect", 1, 3),
    ("shift", 1, 3),
    ("trisect", 2, 3),
    ("trisegment", 2, 2),
    ("iso_triangle_vertex", 1, 2),
    ("iso_triangle_vertex_angle", 1, 2),
    ("eq_triangle", 1, 2),
    ("parallelogram", 1, 3),
    ("square", 2, 2),
    ("circle", 1, 3),
    ("circumcenter", 1, 3),
    ("psquare", 1, 2),
    ("nsquare", 1, 2),
    ("centroid", 4, 3),
    ("s_angle", 1, 2),
    ("eqangle3", 1, 5),
    ("eqratio", 1, 7),
    ("eqratio6", 1, 6),
    ("lconst", 1, 1),
]


def point_name(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if index < len(alphabet):
        return alphabet[index]
    return f"{alphabet[index % len(alphabet)]}{index // len(alphabet)}"


def pick_inputs(rng: random.Random, points: list[str], count: int) -> list[str]:
    if count == 0:
        return []
    if len(points) >= count:
        return rng.sample(points, count)
    return [rng.choice(points) for _ in range(count)]


def random_base(rng: random.Random) -> tuple[str, list[str]]:
    base, points = rng.choice(INITIAL_PROBLEMS)
    points = list(points)
    clauses = [base]
    next_index = len(points)
    for _ in range(rng.randint(0, 4)):
        name, output_count, input_count = rng.choice(
            [c for c in CONSTRUCTIONS if c[2] <= len(points) and c[1] <= 2]
        )
        outputs = [point_name(next_index + i) for i in range(output_count)]
        next_index += output_count
        inputs = pick_inputs(rng, points, input_count)
        clauses.append(f"{' '.join(outputs)} = {name} {' '.join(outputs + inputs)}")
        points.extend(outputs)
    return "; ".join(clauses), points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output", default="generated_aux_constructions.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    rows = []

    for i in range(args.count):
        base_problem, points = random_base(rng)
        name, output_count, input_count = rng.choice(CONSTRUCTIONS)
        next_index = len(points)
        outputs = [point_name(next_index + j) for j in range(output_count)]
        inputs = pick_inputs(rng, points, input_count)
        aux = f"{' '.join(outputs)} = {name} {' '.join(outputs + inputs)}"
        rows.append(
            {
                "id": i + 1,
                "base_problem": base_problem,
                "auxiliary_construction": aux,
                "problem_with_aux": f"{base_problem} | {aux}",
            }
        )

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"generated {len(rows)} rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
