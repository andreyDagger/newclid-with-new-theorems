import math
import unittest
from pathlib import Path

from geometry_drawer import GeometryProblem, parse_text


def constraint_loss(terms):
    return sum(value for key, value in terms.items() if key != "separation")


def solve(text, steps=500, lr=0.04, separation_weight=0.0, init_noise=0.0):
    problem = GeometryProblem(parse_text(text))
    xy, terms = problem.optimize(
        steps=steps,
        lr=lr,
        restarts=1,
        separation_weight=separation_weight,
        seed=7,
        init_noise=init_noise,
    )
    return problem, xy, terms


class CommandConvergenceTests(unittest.TestCase):
    def assert_term_small(self, text, term, threshold=1e-7, **kwargs):
        _, _, terms = solve(text, **kwargs)
        self.assertLess(terms[term], threshold, f"{term} did not converge: {terms}")

    def test_cong_converges(self):
        self.assert_term_small(
            """
            fix a 0 0
            fix b 1 0
            fix c 0 1
            init d 0.2 1.7
            cong a b c d
            """,
            "cong",
        )

    def test_coll_converges(self):
        self.assert_term_small(
            """
            fix a 0 0
            fix b 2 0
            init c 0.8 1.1
            coll a b c
            """,
            "coll",
        )

    def test_cyclic_converges(self):
        self.assert_term_small(
            """
            fix a 1 0
            fix b 0 1
            fix c -1 0
            init d -0.3 -0.8
            cyclic a b c d
            """,
            "cyclic",
            threshold=1e-6,
        )

    def test_eqangle_six_point_form_converges_and_normalizes(self):
        problem, _, terms = solve(
            """
            fix a 1 0
            fix b 0 0
            fix c 0 1
            fix d 1 1
            fix e 0 0
            init f -0.7 0.4
            eqangle a b c d e f
            """,
            steps=1100,
        )
        self.assertLess(terms["eqangle"], 1e-7)
        self.assertIn("eqangle a b b c d e e f", problem.normalized_commands())

    def test_eqangle_eight_point_form_converges(self):
        self.assert_term_small(
            """
            fix a 0 0
            fix b 1 0
            fix c 0 0
            fix d 0 1
            fix e 2 0
            fix f 3 0
            fix g 2 0
            init h 2.7 0.6
            eqangle a b c d e f g h
            """,
            "eqangle",
            threshold=1e-7,
            steps=700,
        )

    def test_midpoint_converges(self):
        self.assert_term_small(
            """
            fix a -1 2
            fix b 3 0
            init m 0 0
            midpoint m a b
            """,
            "midpoint",
        )

    def test_parallel_converges(self):
        self.assert_term_small(
            """
            fix a 0 0
            fix b 1 0
            fix c 0 1
            init d 1.0 1.7
            parallel a b c d
            """,
            "parallel",
        )

    def test_perp_converges(self):
        self.assert_term_small(
            """
            fix a 0 0
            fix b 1 0
            fix c 0 1
            init d 0.8 1.6
            perp a b c d
            """,
            "perp",
        )

    def test_fix_and_init_are_applied(self):
        problem = GeometryProblem(
            parse_text(
                """
                fix a 2 3
                init b -4 5
                """
            )
        )
        xy = problem.initial_tensor(seed=1, noise=0.0)
        coords = problem.coordinates(xy)
        self.assertEqual(coords["a"], (2.0, 3.0))
        self.assertEqual(coords["b"], (-4.0, 5.0))

    def test_separation_penalty_matches_formula(self):
        problem = GeometryProblem(
            parse_text(
                """
                fix a 0 0
                fix b 1 0
                fix c 0 2
                """
            )
        )
        xy = problem.initial_tensor(seed=1, noise=0.0)
        expected = math.exp(-40.0) + math.exp(-160.0) + math.exp(-200.0)
        self.assertAlmostEqual(problem.separation_penalty(xy).item(), expected)


class ExampleTests(unittest.TestCase):
    def test_simple_example_constraints_converge(self):
        commands = Path("examples/simple.txt").read_text(encoding="utf-8")
        _, _, terms = solve(commands, steps=800, lr=0.03)
        self.assertLess(constraint_loss(terms), 1e-6)

    def test_imo_2007_p2_setup_and_angle_check(self):
        commands = Path("examples/imo2007_p2.txt").read_text(encoding="utf-8")
        problem, xy, terms = solve(
            commands,
            steps=800,
            lr=0.03,
        )
        self.assertLess(constraint_loss(terms), 1e-5)
        checks = problem.checks(xy)
        self.assertEqual(len(checks), 1)
        self.assertLess(checks[0][1], 1e-3)

    def test_imo_2008_p1_setup_and_cyclic_checks(self):
        commands = Path("examples/imo2008_p1.txt").read_text(encoding="utf-8")
        problem, xy, terms = solve(
            commands,
            steps=800,
            lr=0.03,
        )
        self.assertLess(constraint_loss(terms), 1e-5)
        checks = problem.checks(xy)
        self.assertEqual(len(checks), 3)
        self.assertLess(max(residual for _, residual in checks), 1e-3)

    def test_imo_2004_p1_setup_and_common_point_check(self):
        commands = Path("examples/imo2004_p1.txt").read_text(encoding="utf-8")
        problem, xy, terms = solve(
            commands,
            steps=1200,
            lr=0.025,
        )
        self.assertLess(constraint_loss(terms), 1e-6)
        checks = problem.checks(xy)
        self.assertEqual(len(checks), 1)
        self.assertLess(checks[0][1], 1e-3)

    def test_imo_2014_p4_setup_and_circumcircle_check(self):
        commands = Path("examples/imo2014_p4.txt").read_text(encoding="utf-8")
        problem, xy, terms = solve(
            commands,
            steps=800,
            lr=0.03,
        )
        self.assertLess(constraint_loss(terms), 1e-8)
        checks = problem.checks(xy)
        self.assertEqual(len(checks), 1)
        self.assertLess(checks[0][1], 1e-6)

    def test_imo_2019_p2_setup_and_cyclic_check(self):
        commands = Path("examples/imo2019_p2.txt").read_text(encoding="utf-8")
        problem, xy, terms = solve(
            commands,
            steps=1200,
            lr=0.02,
        )
        self.assertLess(constraint_loss(terms), 1e-7)
        checks = problem.checks(xy)
        self.assertEqual(len(checks), 1)
        self.assertLess(checks[0][1], 1e-4)


if __name__ == "__main__":
    unittest.main()
