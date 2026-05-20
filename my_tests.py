from newclid import GeometricSolverBuilder, GeometricSolver
from newclid.jgex.problem_builder import JGEXProblemBuilder
from pathlib import Path
from newclid.jgex.formulation import JGEXFormulation
import numpy as np

rng = np.random.default_rng()

rule_to_statement = {
    "pascal": "a b c = triangle a b c; d = on_circum d a b c; e = on_circum e a b c; f = on_circum f a b c; x = on_line x a b, on_line x d e; y = on_line y b c, on_line y e f; z = on_line z c d, on_line z a f ? coll x y z",
    "desargues": "a1 b1 c1 = triangle a1 b1 c1; a2 b2 = segment a2 b2; p = on_line p a1 a2, on_line p b1 b2; c2 = on_line c2 p c1; x = on_line x a1 b1, on_line x a2 b2; y = on_line y a1 c1, on_line y a2 c2; z = on_line z b1 c1, on_line z b2 c2 ? coll x y z",
    "pappus": "a b = segment a b; c = on_line c a b; p q = segment p q; r = on_line r p q; x = on_line x a q, on_line x b p; y = on_line y a r, on_line y c p; z = on_line z b r, on_line z c q ? coll x y z",
    "radical_axes": "a b c = triangle a b c; a1 = on_line a1 b c; b1 = on_line b1 a c; p = on_line p a a1; q = on_line q b b1, on_pline q p a b; p1 = on_line p1 p b1, eqangle3 p1 p c a b c; q1 = on_line q1 q a1, eqangle3 q1 c q b c a; r = on_line r p p1, on_line r q q1; e = on_line e p q, on_line e a c; f = on_line f p q, on_line f b c; t = on_line t a q, on_line t b p; s = on_line s r t, on_line s p q; u = on_line u r t, on_line u a b ? cyclic p p1 q1 q",
    "newton_gauss_line": "a b c = triangle a b c; a1 = on_line a1 b c; b1 = on_line b1 a c; c1 = on_line c1 a b, on_line c1 a1 b1; a2 = midpoint a2 a a1; b2 = midpoint b2 b b1; c2 = midpoint c2 c c1 ? coll a2 b2 c2",
     }
for name, statement in rule_to_statement.items():
    problem_setup = JGEXProblemBuilder(rng=rng).with_problem_from_txt(
      statement
    ).build()

    solver: GeometricSolver = GeometricSolverBuilder().build(problem_setup)

    success = solver.run()

    assert success

