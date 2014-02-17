// constrained optimization through projection
package optimization

import (
	"math"
)

const (
	dbl_max     = math.MaxFloat64
	dbl_epsilon = 2.22044604925031308085e-16
)

// TODO: COW(copy on write)? anytime I need to write, I just copy. But we can do
// better to avoid the copy if we know the reference count is one?
type Point struct {
	Factor float64
	Dense  []float64
}

func DensePoint(a []float64) Point {
	return Point{Factor: 1.0, Dense: a}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(x float64, y float64) float64 {
	if x < y {
		return x
	}
	return y
}

func imin(x int, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x float64, y float64) float64 {
	if x > y {
		return x
	}
	return y
}

func imax(x int, y int) int {
	if x > y {
		return x
	}
	return y
}

func (a *Point) Size() int {
	if a.Dense != nil {
		return len(a.Dense)
	}
	return 0
}

func (a *Point) Scale(x float64) Point {
	return Point{Factor: a.Factor * x,
		Dense: a.Dense}
}

func (a *Point) AbsMax() float64 {
	if a.Factor == 0 {
		return 0
	}
	m := 0.0
	if a.Dense != nil {
		for _, v := range a.Dense {
			if abs(v) > m {
				m = abs(v)
			}
		}
	}
	return m * abs(a.Factor)
}

func (a *Point) SquareSum() float64 {
	if a.Factor == 0 {
		return 0
	}
	m := 0.0
	if a.Dense != nil {
		for _, v := range a.Dense {
			m += v * v
		}
	}
	return m * a.Factor * a.Factor
}

func plusImpl(dest *Point, ds []Point) {
	if len(ds) == 0 {
		return
	}
	if dest.Size() == 0 {
		dest.Factor = 0.0
	}
	is_alloc := false
	if len(ds) > 0 {
		c := 0
		if dest.Factor == 0.0 {
			*dest = ds[0]
			is_alloc = false
			c = 1
		}
		if len(ds) > c && is_alloc == false {
			t := Point{Factor: 1, Dense: make([]float64, len(ds[0].Dense))}
			if dest.Dense != nil {
				for x, v := range dest.Dense {
					t.Dense[x] = v * dest.Factor
				}
			}
			*dest = t
			is_alloc = true
		}
		for ; c < len(ds); c++ {
			for x, v := range ds[c].Dense {
				dest.Dense[x] += ds[c].Factor * v
			}
		}
	}
	return
}

func Sum(vs ...Point) (ret Point) {
	plusImpl(&ret, vs)
	return ret
}

func PlusAssign(d *Point, vs ...Point) {
	plusImpl(d, vs)
}

func (a *Point) InnerProd(b Point) float64 {
	if a.Size() == 0 || b.Size() == 0 {
		return 0
	}
	s := 0.0
	for x, v := range a.Dense {
		s += v * b.Dense[x]
	}
	return s * a.Factor * b.Factor
}

type Problem struct {
	Value                func(Point) float64
	Gradient             func(Point) Point
	ProjectFunc          func(Point) Point
	DirectionProjectFunc func(Point, Point) Point
	GradientProjectFunc  func(Point, Point) Point
	LinePoint            Point
	LineDirection        Point
	LineUpdateCallback   func(Point, Point)
	LineValueFunc        func(float64) float64
	LineGradientFunc     func(float64) float64
}

func (problem *Problem) Project(p Point) Point {
	if problem.ProjectFunc != nil {
		return problem.ProjectFunc(p)
	}
	return p
}

func (problem *Problem) DirectionProject(p Point, d Point) Point {
	if problem.DirectionProjectFunc != nil {
		return problem.DirectionProjectFunc(p, d)
	}
	return d
}

func (problem *Problem) GradientProject(p Point, g Point) Point {
	if problem.GradientProjectFunc != nil {
		return problem.GradientProjectFunc(p, g)
	}
	return g
}

func (problem *Problem) LineUpdate(p Point, d Point) {
	if problem.LineUpdateCallback != nil {
		problem.LineUpdateCallback(p, d)
	}
	problem.LinePoint = p
	problem.LineDirection = d
}

func (problem *Problem) LineValue(alpha float64) float64 {
	if problem.LineValueFunc != nil {
		return problem.LineValueFunc(alpha)
	}
	p := Sum(problem.LinePoint, problem.LineDirection.Scale(alpha))
	p = problem.Project(p)
	return problem.Value(p)
}

func (problem *Problem) LineGradient(alpha float64) float64 {
	if problem.LineGradientFunc != nil {
		return problem.LineGradientFunc(alpha)
	}
	p := Sum(problem.LinePoint, problem.LineDirection.Scale(alpha))
	p = problem.Project(p)
	g := problem.GradientProject(p, problem.Gradient(p))
	return g.InnerProd(problem.LineDirection)
}

type LineSearch interface {
	Search(*Problem, float64) float64
}

type FirstLinePoint struct {
	problem      *Problem
	point        float64
	value        float64
	has_value    bool
	gradient     float64
	has_gradient bool
}

func (a *FirstLinePoint) Point() float64 {
	return a.point
}

func (a *FirstLinePoint) Value() float64 {
	if !a.has_value {
		a.value = a.problem.LineValue(a.point)
		a.has_value = true
	}
	return a.value
}

func (a *FirstLinePoint) Gradient() float64 {
	if !a.has_gradient {
		a.gradient = a.problem.LineGradient(a.point)
		a.has_gradient = true
	}
	return a.gradient
}

func cubicMin(ax, ay, ag, bx, by, bg float64) float64 {
	dx := bx - ax
	if dx == 0 {
		return dbl_max
	}
	dy := by - ay
	Bg := dy / dx
	T := (bg - Bg) + (ag - Bg)
	Z := T - Bg
	q := Z*Z - ag*bg
	if q < 0 {
		return dbl_max
	}
	q = math.Sqrt(q)
	if dx < 0 {
		q = -q
	}
	if (Z+ag)*q < 0 {
		if abs(Z+ag-q) > dbl_epsilon {
			return ax + dx*ag/(Z+ag-q)
		}
	}
	if abs(T) < dbl_epsilon {
		return dbl_max
	}
	return ax + dx*(Z+ag+q)/(3*T)
}

func quadMin(ax, ay, ag, bx, by float64) float64 {
	if ax == bx {
		return dbl_max
	}
	Bg := (by - ay) / (bx - ax)
	gg := (Bg - ag) / (bx - ax)
	if gg < dbl_epsilon {
		return dbl_max
	}
	return ax - ag/(2*gg)
}

func interpolate(ax, ay, ag, bx, by, bg float64) float64 {
	x := cubicMin(ax, ay, ag, bx, by, bg)
	lx := ax
	rx := bx
	if lx > rx {
		lx, rx = rx, lx
	}
	cchk := .2 * (rx - lx)
	if x >= lx+cchk && x <= rx-cchk {
		return x
	}
	x = quadMin(ax, ay, ag, bx, by)
	qchk := (rx - lx) * .01
	if x >= lx+qchk && x <= rx-qchk {
		return x
	}
	return .5 * (lx + rx)
}

func cubicInterpolate(ax, ay, ag, bx, by, bg, plx, prx float64) float64 {
	x := cubicMin(ax, ay, ag, bx, by, bg)
	lx := ax
	rx := bx
	if lx > rx {
		lx, rx = rx, lx
	}
	cchk := .2 * (rx - lx)
	if x < rx && x > lx {
		if x >= lx+cchk && x <= rx-cchk {
			return x
		}
		if prx > rx && x < lx+cchk {
			return x
		}
		if plx < lx && x > rx-cchk {
			return x
		}
	}
	x = quadMin(ax, ay, ag, bx, by)
	qchk := (rx - lx) * .01
	if x < rx && x > lx {
		if x >= lx+qchk && x <= rx-qchk {
			return x
		}
		if prx > rx && x < lx+qchk {
			return x
		}
		if plx < lx && x > rx-qchk {
			return x
		}
	}
	return .5 * (lx + rx)
}

func zoom(p *Problem, c1, c2 float64,
	o, al, ah FirstLinePoint,
	max_iter int) FirstLinePoint {
	a := al
	plx := -dbl_max
	prx := dbl_max
	for iter := 0; iter < max_iter; iter++ {
		a = FirstLinePoint{
			problem: p,
			point: cubicInterpolate(al.Point(), al.Value(), al.Gradient(),
				ah.Point(), ah.Value(), ah.Gradient(),
				plx, prx)}
		plx = al.Point()
		prx = ah.Point()
		if plx > prx {
			plx, prx = prx, plx
		}
		if a.Point() == math.Nextafter(al.Point(), a.Point()) ||
			a.Point() == math.Nextafter(ah.Point(), a.Point()) {
			break
		}
		if a.Value() > o.Value()+c1*a.Point()*o.Gradient() || a.Value() >= al.Value() {
			ah = a
		} else {
			if abs(a.Gradient()) <= -c2*o.Gradient() {
				break
			}
			if a.Gradient()*(ah.Point()-al.Point()) >= 0 {
				ah = al
			}
			al = a
		}
	}
	if a.Value() < al.Value() && a.Value() < ah.Value() {
		return a
	}
	if al.Value() < ah.Value() {
		return al
	}
	return ah
}

type BacktrackingLineSearch struct{}

func (s *BacktrackingLineSearch) Search(p *Problem, alpha float64) float64 {
	sufficient_decrease := .25 // should be in (0, .5),
	backstep := .7             // should be in (0, 1)
	alpha_epsilon := 1.e-100

	if alpha == 0 {
		alpha = 1.0
	}
	O := FirstLinePoint{problem: p, point: 0.0}
	T := FirstLinePoint{problem: p, point: alpha}
	for T.Value() > O.Value()+sufficient_decrease*T.Point()*O.Gradient() {
		T = FirstLinePoint{problem: p, point: T.Point() * backstep}
		if T.Point() <= alpha_epsilon {
			break
		}
	}
	return T.Point()
}

type StrongWolfeLineSearch struct{}

func (s *StrongWolfeLineSearch) Search(p *Problem, alpha float64) float64 {
	alpha_max := 1.e99
	max_iter := 10
	zoom_max_iter := 10
	sufficient_decrease := 1.e-4
	curvature := 0.1

	O := FirstLinePoint{problem: p, point: 0}
	if alpha == 0 {
		x_infty := p.LinePoint.AbsMax()
		if x_infty > 0 {
			g := p.Gradient(p.LinePoint)
			g_infty := g.AbsMax()
			alpha = .01 * x_infty / g_infty
		}
	}
	if alpha == 0 {
		alpha = 1.0
	}
	A := FirstLinePoint{problem: p, point: alpha}
	Ap := O
	for iter := 0; iter < max_iter; iter++ {
		if A.Value() > O.Value()+sufficient_decrease*A.Point()*O.Gradient() || (A.Value() >= Ap.Value() && iter > 0) {
			X := zoom(p, sufficient_decrease, curvature, O, Ap, A, zoom_max_iter)
			return X.Point()
		}
		if abs(A.Gradient()) <= -curvature*O.Gradient() {
			return A.Point()
		}
		if A.Gradient() >= 0 {
			X := zoom(p, sufficient_decrease, curvature, O, A, Ap, zoom_max_iter)
			return X.Point()
		}
		if A.Point() >= alpha_max {
			break
		}
		x := cubicMin(Ap.Point(), Ap.Value(), Ap.Gradient(), A.Point(), A.Value(), A.Gradient())
		Ap = A
		if x <= A.Point() || x >= alpha_max {
			x = A.Point() * 2.0 * (float64(iter) + 2.0)
		}
		A = FirstLinePoint{problem: p, point: x}
	}
	if A.Value() < O.Value() {
		return A.Point()
	}
	return O.Point()
}

const (
	check_break_rollback = -2
	check_break_keep     = -1
	check_break_max      = 0
	check_rollback       = 1
	check_forward        = 2
)

type Solver interface {
	Solve(*Problem, Point) (float64, Point)
	Check(float64, float64) int
}

type SolverBase struct {
}

func (solver *SolverBase) Check(c, pc float64) int {
	if c > pc {
		return check_break_rollback
	}
	if c == pc {
		return check_break_keep
	}
	return check_forward
}

type GradientDescentSolver struct {
	SolverBase
	Line    LineSearch
	MaxIter int
}

func (solver *GradientDescentSolver) Solve(problem *Problem, p Point) (float64, Point) {
	line_search := solver.Line
	if line_search == nil {
		line_search = &StrongWolfeLineSearch{}
	}
	problem.Project(p)
	cost := problem.Value(p)
	pre_dg := 0.0
	alpha := 0.0
	max_iter := solver.MaxIter
	if max_iter == 0 {
		max_iter = 30
	}
	for iter := 0; iter < max_iter; iter++ {
		g := problem.Gradient(p)
		g = problem.GradientProject(p, g)
		d := g.Scale(-1)
		d = problem.DirectionProject(p, d)
		dg := d.InnerProd(g)
		if dg >= 0 {
			break
		}
		problem.LineUpdate(p, d)
		if iter > 0 {
			alpha *= pre_dg / dg
		}
		pre_dg = dg
		alpha = line_search.Search(problem, alpha)
		pn := Sum(p, d.Scale(alpha))
		pn = problem.Project(pn)
		nc := problem.LineValue(alpha)
		result := solver.Check(nc, cost)
		if result == check_break_rollback || result == check_rollback {
			break
		}
		cost = nc
		p = pn
		if result < check_break_max {
			break
		}
	}
	return cost, p
}

type ConjugateGradientBeta interface {
	Beta(alpha float64, direction, previous_gradient, point, gradient Point) float64
}

type ConjugateGradientBetaPRP struct{}

func (s *ConjugateGradientBetaPRP) Beta(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g - pg> / <pg, pg>
	den := previous_gradient.SquareSum()
	num := gradient.SquareSum() - gradient.InnerProd(previous_gradient)
	if den > 0 && num > 0 {
		return num / den
	}
	return 0.0
}

type ConjugateGradientBetaFR struct{}

func (s *ConjugateGradientBetaFR) Beta(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g> / <pg, pg>
	num := gradient.SquareSum()
	den := previous_gradient.SquareSum()
	if den > 0 {
		return num / den
	}
	return 0
}

type ConjugateGradientBetaHS struct{}

func (s *ConjugateGradientBetaHS) Beta(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g - pg> / <d, g - pg>
	num := gradient.SquareSum() - gradient.InnerProd(previous_gradient)
	den := direction.InnerProd(gradient) - direction.InnerProd(previous_gradient)
	if den != 0 {
		return num / den
	}
	return 0
}

type ConjugateGradientBetaDY struct{}

func (s *ConjugateGradientBetaDY) Beta(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g> / <g - pg, d>
	num := gradient.SquareSum()
	den := gradient.InnerProd(direction) - previous_gradient.InnerProd(direction)
	if den != 0 {
		return num / den
	}
	return 0
}

type ConjugateGradientBetaDescent struct{}

func (s *ConjugateGradientBetaDescent) Beta(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// y = g - pg
	eta2 := 1.e-4
	g_g := gradient.SquareSum()
	g_p := gradient.InnerProd(previous_gradient)
	g_d := gradient.InnerProd(direction)
	p_p := previous_gradient.SquareSum()
	p_d := previous_gradient.InnerProd(direction)
	d_d := direction.SquareSum()
	yy := g_g + p_p - 2*g_p
	dy := g_d - p_d
	dg := g_d
	gy := g_g - g_p
	dd := d_d
	pp := p_p
	b := (gy - 2*yy/dy*dg) / dy
	t := -1. / math.Sqrt(dd*min(eta2, pp))
	return max(t, b)
}

type ConjugateGradientSolver struct {
	SolverBase
	Line    LineSearch
	MaxIter int
	Beta    ConjugateGradientBeta
}

func (solver *ConjugateGradientSolver) Solve(problem *Problem, p Point) (float64, Point) {
	line_search := solver.Line
	beta := solver.Beta
	if line_search == nil {
		line_search = &StrongWolfeLineSearch{}
	}
	if beta == nil {
		beta = &ConjugateGradientBetaPRP{}
	}
	cost := problem.Value(p)
	g := problem.GradientProject(p, problem.Gradient(p))
	d := g.Scale(-1)
	d = problem.DirectionProject(p, d)
	pre_dg := 0.0
	alpha := 0.0
	max_iter := solver.MaxIter
	if max_iter == 0 {
		max_iter = 30
	}
	pg := Point{}
	for iter := 0; iter < max_iter; iter++ {
		dg := d.InnerProd(g)
		if dg >= 0 {
			d = g.Scale(-1)
			d = problem.DirectionProject(p, d)
			dg = d.InnerProd(g)
		}
		if dg >= 0 {
			break
		}
		problem.LineUpdate(p, d)
		if iter > 0 {
			alpha = pre_dg / dg
		}
		pre_dg = dg
		alpha = line_search.Search(problem, alpha)
		pn := Sum(p, d.Scale(alpha))
		problem.Project(pn)
		nc := problem.LineValue(alpha)
		result := solver.Check(nc, cost)
		if result == check_break_rollback {
			break
		}
		if result == check_forward || result == check_break_keep {
			cost = nc
			p = pn
		}
		if iter+1 >= max_iter || result < check_break_max {
			break
		}
		if result == check_forward {
			pg = g
			g = problem.GradientProject(p, problem.Gradient(p))
			// Note: line search is not exact, so disable powell's
			// restart criteria.
			if false && abs(g.InnerProd(pg)) >= .2*g.SquareSum() {
				d = g.Scale(-1)
			} else {
				b := beta.Beta(alpha, d, pg, p, g)
				d = Sum(d.Scale(b), g.Scale(-1))
			}
		} else {
			// restart
			g = pg
			d = g.Scale(-1)
		}
	}
	return cost, p
}

type LmBFGSSolver struct {
	SolverBase
	Line    LineSearch
	MaxIter int
	Recent  int
}

func (solver *LmBFGSSolver) Solve(problem *Problem, p Point) (float64, Point) {
	line_search := solver.Line
	if line_search == nil {
		line_search = &StrongWolfeLineSearch{}
	}
	recent := solver.Recent
	if recent == 0 {
		recent = 5
	}
	alpha := 0.0
	p = problem.Project(p)
	c := problem.Value(p)
	pre_c := c
	pre_is_rollback := false
	recent_start := 0
	d := Point{}
	g := problem.GradientProject(p, problem.Gradient(p))
	logs_rho := make([]float64, recent)
	logs_alpha := make([]float64, recent)
	logs_dg := make([]Point, recent)
	logs_dx := make([]Point, recent)
	max_iter := solver.MaxIter
	if max_iter == 0 {
		max_iter = 30
	}
	for iter := 0; iter < max_iter; iter++ {
		if iter > 0 {
			k := (iter - 1) % recent
			v := logs_dg[k].InnerProd(logs_dx[k])
			if abs(v) > dbl_epsilon {
				logs_rho[k] = 1. / v
			} else {
				logs_rho[k] = 0.
			}
		}
		// calculation direction d
		if pre_is_rollback || iter <= recent_start {
			d = g.Scale(-1)
		} else {
			q := g
			km := imax(iter-recent, recent_start)
			for i := iter - 1; i >= km; i-- {
				k := i % recent
				a := logs_rho[k] * q.InnerProd(logs_dx[k])
				logs_alpha[k] = a
				q = Sum(q, logs_dg[k].Scale(-a))
			}
			num := logs_dx[(iter-1)%recent].InnerProd(logs_dg[(iter-1)%recent])
			den := logs_dg[(iter-1)%recent].SquareSum()
			if den > 0 {
				d = q.Scale(num / den)
			} else {
				d = q
			}
			for i := km; i < iter; i++ {
				k := i % recent
				b := logs_rho[k] * d.InnerProd(logs_dg[k])
				d = Sum(d, logs_dx[k].Scale(logs_alpha[k]-b))
			}
			d.Scale(-1)
		}
		dg := d.InnerProd(g)
		if dg >= 0 {
			d = g.Scale(-1)
			dg = -g.SquareSum()
			recent_start = iter
		}
		if dg >= 0 {
			break
		}
		d = problem.DirectionProject(p, d)
		problem.LineUpdate(p, d)
		if iter > 0 {
			a2 := 2 * (c - pre_c) / dg
			alpha = min(1.0, 1.01*a2)
			if alpha <= 0 {
				alpha = 1.0
			}
		}
		pre_c = c
		alpha = line_search.Search(problem, alpha)
		nc := problem.LineValue(alpha)
		result := solver.Check(nc, c)
		if result == check_break_rollback {
			break
		}
		pp := p
		if result == check_break_keep || result == check_forward {
			c = nc
			p = Sum(p, d.Scale(alpha))
			p = problem.Project(p)
		}
		if result < check_break_max {
			break
		}
		if iter+1 >= max_iter {
			break
		}
		if result == check_forward {
			k := iter % recent
			logs_dx[k] = Sum(p, pp.Scale(-1))
			pg := g
			g = problem.GradientProject(p, problem.Gradient(p))
			logs_dg[k] = Sum(g, pg.Scale(-1))
		} else {
			if iter == recent_start {
				break
			}
			recent_start = iter + 1
		}
		pre_is_rollback = (result == check_rollback)
	}
	return c, p
}
