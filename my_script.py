from newclid import GeometricSolverBuilder, GeometricSolver
from newclid.jgex.problem_builder import JGEXProblemBuilder
from pathlib import Path
from newclid.jgex.formulation import JGEXFormulation
import numpy as np

# Set the random generator
rng = np.random.default_rng()

# statement = "a = free a; b = free b; c = on_line c a b; p = free p; q = free q; r = on_line r p q; x = on_line x a q, on_line x b p; y = on_line y a r, on_line y c p; z = on_line z b r, on_line z c q ? coll x y z"
statement = "a b c = triangle a b c; a1 = on_line a1 b c; b1 = on_line b1 a c; p = on_line p a a1; q = on_line q b b1, on_pline q p a b; p1 = on_line p1 p b1, eqangle3 p1 p c a b c; q1 = on_line q1 q a1, eqangle3 q1 c q b c a; r = on_line r p p1, on_line r q q1; e = on_line e p q, on_line e a c; f = on_line f p q, on_line f b c; t = on_line t a q, on_line t b p; s = on_line s r t, on_line s p q; u = on_line u r t, on_line u a b ? cyclic p p1 q1 q"
# Build the problem setup from JGEX string
problem_setup = JGEXProblemBuilder(rng=rng).with_problem_from_txt(
  statement
).build()

# We now build the solver on the problem
solver: GeometricSolver = GeometricSolverBuilder().build(problem_setup)

# And run the solver
success = solver.run()

if success:
    print("Successfuly solved the problem! Proof:")
    solver.write_all_outputs(out_folder_path=Path('./output'),jgex_problem=JGEXFormulation.from_text(statement))
else:
    print("Failed to solve the problem...")

print(f"Run infos {solver.run_infos}")

