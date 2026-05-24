# Geometry Drawer

Numerical PyTorch drawer for small formal geometry descriptions.

Supported constraint commands:

```text
cong a b c d          # |AB| = |CD|
coll a b c            # A, B, C are collinear
cyclic a b c d        # A, B, C, D are cyclic
dists_equation v1 a1 b1 ... vn an bn target
                       # v1*|A1B1| + ... + vn*|AnBn| = target
eqangle a b c d e f   # angle ABC equals angle DEF, unoriented
eqangle a b c d e f g h
                       # oriented angle between lines AB and CD equals EF and GH
angles_equation v1 a1 b1 c1 ... vn an bn cn target
                       # v1*angle A1B1C1 + ... + vn*angle AnBnCn = target
eqratio a b c d e f g h
                       # minimizes (|AB|/|CD| - |EF|/|GH|)^2
midpoint m a b        # M is the midpoint of AB
parallel a b c d      # AB is parallel to CD
para a b c d          # alias for parallel
perp a b c d          # AB is perpendicular to CD
convex_polygon a1 a2 ... an
                       # zero loss when points are convex in this cyclic order
x = intersect a b c d  # X is the intersection of lines AB and CD
h = orthocenter a b c  # H is the orthocenter of triangle ABC
i = incenter a b c     # I is the incenter of triangle ABC
ia = excenter a b c    # IA is the A-excenter, opposite the first vertex
fix a x y             # fixed coordinate
init a x y            # initial coordinate only
check_eqangle ...     # report-only check, not optimized
check_cyclic a b c d  # report-only cyclic check, not optimized
```

The anti-collapse penalty is exactly

```text
sum_{i < j} exp(-40 * dist(point_i, point_j)^2)
```

It is added with `--separation-weight`, which defaults to `0.001`, so exact
geometry constraints dominate while coincident points are still discouraged.

Run:

```powershell
python geometry_drawer.py examples/simple.txt --threads 8 --plot out/simple.png
python geometry_drawer.py examples/imo2004_p1.txt --steps 1200 --restarts 1 --init-noise 0 --plot out/imo2004_p1.png
python geometry_drawer.py examples/imo2007_p2.txt --steps 3000 --restarts 1 --init-noise 0 --plot out/imo2007_p2.png
python geometry_drawer.py examples/imo2008_p1.txt --steps 1200 --restarts 1 --init-noise 0 --plot out/imo2008_p1.png
python geometry_drawer.py examples/imo2014_p4.txt --steps 800 --restarts 1 --init-noise 0 --plot out/imo2014_p4.png
python geometry_drawer.py examples/imo2019_p2.txt --steps 1200 --restarts 1 --init-noise 0 --plot out/imo2019_p2.png
python -m unittest discover -s tests
```

Independent restarts run in parallel with `--threads`. When more than one
thread is used, a `rich.Live` table shows each restart's status, current step,
loss, constraint loss, separation penalty, and best loss. Optimization stops
the remaining workers as soon as one restart reaches `--success-loss` constraint
loss; pass a negative value to disable early stopping.

Constructive assignments such as `h = orthocenter a b c` also initialize the
target point from the formula before optimization, unless that point is fixed.

The output contains optimized coordinates and the normalized commands. Six-point
`eqangle a b c d e f` is printed as eight-point
`eqangle a b b c d e e f`.
