// constrained optimization through projection
package optimization

import (
	"container/list"
	"fmt"
	"math"
)

const (
	BreakRollback    = -2
	BreakKeep        = -1
	BreakMax         = 0
	Rollback         = 1
	Forward          = 2
	dbl_epsilon      = 2.22044604925031308085e-16
	point_cache_size = 4
)

func CheckResultString(result int) string {
	if result == BreakRollback {
		return "BreakRollback"
	}
	if result == BreakKeep {
		return "BreakKeep"
	}
	if result == Rollback {
		return "Rollback"
	}
	if result == Forward {
		return "Forward"
	}
	return fmt.Sprintf("%v", result)
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

type ProblemLineCache struct {
	point    float64
	value    *float64
	gradient *float64
}

type ProblemPointCache struct {
	point    Point
	value    *float64
	gradient *Point
}

type Problem struct {
	LinePoint            Point
	LineDirection        Point
	ValueFunc            func(Point) float64
	GradientFunc         func(Point) Point
	ValueAndGradientFunc func(Point) (float64, Point)
	ProjectFunc          func(Point) Point
	DirectionProjectFunc func(Point, Point) Point
	GradientProjectFunc  func(Point, Point) Point
	LineUpdateFunc       func(Point, Point)
	LineValueFunc        func(float64) float64
	LineGradientFunc     func(float64) float64
	NumValue             int
	NumGradient          int
	NumProject           int
	NumDirectionProject  int
	NumGradientProject   int
	NumLineValue         int
	NumLineUpdate        int
	NumLineGradient      int
	line_cache           map[float64]*ProblemLineCache
	point_cache          list.List
}

func (problem *Problem) Value(p Point) float64 {
	c := (*ProblemPointCache)(nil)
	e := (*list.Element)(nil)
	for e = problem.point_cache.Front(); e != nil; e = e.Next() {
		c = e.Value.(*ProblemPointCache)
		if c.point.Equal(p) {
			if c.value != nil {
				problem.point_cache.MoveToFront(e)
				return *c.value
			}
			break
		} else {
			c = nil
		}
	}
	problem.NumValue++
	f := 0.0
	g := (*Point)(nil)
	if problem.ValueFunc != nil {
		f = problem.ValueFunc(p)
	} else {
		fv, gv := problem.ValueAndGradientFunc(p)
		problem.NumGradient++
		g = &gv
		f = fv
	}
	if point_cache_size > 0 {
		if c != nil {
			c.value = &f
			if g != nil {
				c.gradient = g
			}
			problem.point_cache.MoveToFront(e)
		} else {
			problem.point_cache.PushFront(
				&ProblemPointCache{
					point:    p,
					value:    &f,
					gradient: g})
			if problem.point_cache.Len() > point_cache_size {
				b := problem.point_cache.Back()
				problem.point_cache.Remove(b)
			}
		}
	}
	return f
}

func (problem *Problem) Gradient(p Point) Point {
	c := (*ProblemPointCache)(nil)
	e := (*list.Element)(nil)
	for e = problem.point_cache.Front(); e != nil; e = e.Next() {
		c = e.Value.(*ProblemPointCache)
		if c.point.Equal(p) {
			if c.gradient != nil {
				problem.point_cache.MoveToFront(e)
				return *c.gradient
			}
			break
		} else {
			c = nil
		}
	}
	problem.NumGradient++
	f := (*float64)(nil)
	g := (*Point)(nil)
	if problem.GradientFunc != nil {
		gv := problem.GradientFunc(p)
		g = &gv
	} else {
		problem.NumValue++
		fv, gv := problem.ValueAndGradientFunc(p)
		g = &gv
		f = &fv
	}
	if point_cache_size > 0 {
		if c != nil {
			if f != nil {
				c.value = f
			}
			c.gradient = g
			problem.point_cache.MoveToFront(e)
		} else {
			problem.point_cache.PushFront(
				&ProblemPointCache{
					point:    p,
					value:    f,
					gradient: g})
			if problem.point_cache.Len() > point_cache_size {
				problem.point_cache.Remove(problem.point_cache.Back())
			}
		}
	}
	return *g
}

func (problem *Problem) Project(p Point) Point {
	problem.NumProject++
	if problem.ProjectFunc != nil {
		return problem.ProjectFunc(p)
	}
	return p
}

func (problem *Problem) DirectionProject(p Point, d Point) Point {
	problem.NumDirectionProject++
	if problem.DirectionProjectFunc != nil {
		return problem.DirectionProjectFunc(p, d)
	}
	return d
}

func (problem *Problem) GradientProject(p Point, g Point) Point {
	problem.NumGradientProject++
	if problem.GradientProjectFunc != nil {
		return problem.GradientProjectFunc(p, g)
	}
	return g
}

func (problem *Problem) LineUpdate(p Point, d Point) {
	problem.NumLineUpdate++
	if problem.LineUpdateFunc != nil {
		problem.LineUpdateFunc(p, d)
	}
	problem.LinePoint = p
	problem.LineDirection = d
	problem.line_cache = make(map[float64]*ProblemLineCache)
}

func (problem *Problem) LineValue(alpha float64) float64 {
	c := problem.line_cache[alpha]
	if c == nil {
		c = &ProblemLineCache{}
		problem.line_cache[alpha] = c
	}
	if c.value != nil {
		return *c.value
	}
	problem.NumLineValue++
	v := 0.0
	if problem.LineValueFunc != nil {
		v = problem.LineValueFunc(alpha)
	} else {
		p := Sum(problem.LinePoint, problem.LineDirection.Scale(alpha))
		p = problem.Project(p)
		v = problem.Value(p)
	}
	c.value = &v
	return v
}

func (problem *Problem) LineGradient(alpha float64) float64 {
	c := problem.line_cache[alpha]
	if c == nil {
		c = &ProblemLineCache{}
		problem.line_cache[alpha] = c
	}
	if c.gradient != nil {
		return *c.gradient
	}
	problem.NumLineGradient++
	v := 0.0
	if problem.LineGradientFunc != nil {
		v = problem.LineGradientFunc(alpha)
	} else {
		p := Sum(problem.LinePoint, problem.LineDirection.Scale(alpha))
		p = problem.Project(p)
		g := problem.GradientProject(p, problem.Gradient(p))
		v = g.InnerProd(problem.LineDirection)
	}
	c.gradient = &v
	return v
}

type DerivativeLinePoint struct {
	problem  *Problem
	point    float64
	value    *float64
	gradient *float64
}

func (a *DerivativeLinePoint) Point() float64 {
	return a.point
}

func (a *DerivativeLinePoint) Value() float64 {
	if a.value == nil {
		v := a.problem.LineValue(a.point)
		a.value = &v
	}
	return *a.value
}

func (a *DerivativeLinePoint) Gradient() float64 {
	if a.gradient == nil {
		v := a.problem.LineGradient(a.point)
		a.gradient = &v
	}
	return *a.gradient
}

func cubicMin(ax, ay, ag, bx, by, bg float64) float64 {
	dx := bx - ax
	if dx == 0 {
		return math.MaxFloat64
	}
	dy := by - ay
	Bg := dy / dx
	T := (bg - Bg) + (ag - Bg)
	Z := T - Bg
	q := Z*Z - ag*bg
	if q < 0 {
		return math.MaxFloat64
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
		return math.MaxFloat64
	}
	return ax + dx*(Z+ag+q)/(3*T)
}

func quadMin(ax, ay, ag, bx, by float64) float64 {
	if ax == bx {
		return math.MaxFloat64
	}
	Bg := (by - ay) / (bx - ax)
	gg := (Bg - ag) / (bx - ax)
	if gg < dbl_epsilon {
		return math.MaxFloat64
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
	o, al, ah DerivativeLinePoint,
	max_iter int) DerivativeLinePoint {
	a := al
	plx := -math.MaxFloat64
	prx := math.MaxFloat64
	for iter := 0; iter < max_iter; iter++ {
		a = DerivativeLinePoint{
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

func BacktrackingLineSearch(p *Problem, alpha float64) float64 {
	const sufficient_decrease = .25 // should be in (0, .5),
	const backstep = .7             // should be in (0, 1)
	const alpha_epsilon = 1.e-100

	if alpha == 0 {
		alpha = 1.0
	}
	O := DerivativeLinePoint{problem: p, point: 0.0}
	T := DerivativeLinePoint{problem: p, point: alpha}
	for T.Value() > O.Value()+sufficient_decrease*T.Point()*O.Gradient() {
		T = DerivativeLinePoint{problem: p, point: T.Point() * backstep}
		if T.Point() <= alpha_epsilon {
			break
		}
	}
	return T.Point()
}

func StrongWolfeLineSearch(p *Problem, alpha float64) float64 {
	const alpha_max = math.MaxFloat64
	const max_iter = 10
	const zoom_max_iter = 10
	const sufficient_decrease = 1.e-4
	const curvature = 0.1

	O := DerivativeLinePoint{problem: p, point: 0}
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
	A := DerivativeLinePoint{problem: p, point: alpha}
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
		A = DerivativeLinePoint{problem: p, point: x}
	}
	if A.Value() < O.Value() {
		return A.Point()
	}
	return O.Point()
}

type SolverIterationStats struct {
	Iteration             int
	NumFunction           int
	NumGradient           int
	AccumulateNumFunction int
	AccumulateNumGradient int
	X                     Point
	Y                     float64
	G                     Point
	CheckResult           int
}

type Solver interface {
	Check(*Problem, SolverIterationStats, Point, float64) int
	Solve(*Problem, Point) (Point, float64)
	Init(map[string]interface{})
	Log(int, string)
	Logf(int, string, ...interface{})
	LogIterationStats(SolverIterationStats)
}

type SolverBase struct {
	CheckFunc             func(*Problem, SolverIterationStats, Point, float64) int
	MaxIter               int
	Line                  func(*Problem, float64) float64
	LogFunc               func(int, string)
	LogIterationStatsFunc func(SolverIterationStats)
}

func (solver *SolverBase) Check(problem *Problem, stats SolverIterationStats, xp Point, yp float64) int {
	if solver.CheckFunc != nil {
		return solver.CheckFunc(problem, stats, xp, yp)
	}
	y := stats.Y
	if y > yp {
		return BreakRollback
	}
	if y == yp {
		return BreakKeep
	}
	return Forward
}

func (solver *SolverBase) Log(level int, message string) {
	if solver.LogFunc != nil {
		solver.LogFunc(level, message)
	}
	return
}

func (solver *SolverBase) Logf(level int, format string, vs ...interface{}) {
	solver.Log(level, fmt.Sprintf(format, vs...))
}

func (solver *SolverBase) LogIterationStats(stats SolverIterationStats) {
	if solver.LogIterationStatsFunc != nil {
		solver.LogIterationStatsFunc(stats)
	} else {
		solver.Logf(100, "iter=%v y=%v #f=%v/%v #g=%v/%v result=%s",
			stats.Iteration, stats.Y, stats.NumFunction, stats.AccumulateNumFunction, stats.NumGradient, stats.AccumulateNumGradient,
			CheckResultString(stats.CheckResult))
	}
	return
}

func (solver *SolverBase) Init(kwds map[string]interface{}) {
	solver.MaxIter = 0
	if v, ok := kwds["MaxIter"]; ok {
		solver.MaxIter = v.(int)
	} else {
		solver.MaxIter = 30
	}
	if v, ok := kwds["LineSearch"]; ok {
		solver.Line = v.(func(*Problem, float64) float64)
	} else {
		solver.Line = StrongWolfeLineSearch
	}
	if v, ok := kwds["CheckFunc"]; ok {
		solver.CheckFunc = v.(func(*Problem, SolverIterationStats, Point, float64) int)
	}
	if v, ok := kwds["LogFunc"]; ok {
		solver.LogFunc = v.(func(int, string))
	}
	if v, ok := kwds["LogIterationStatsFunc"]; ok {
		solver.LogIterationStatsFunc = v.(func(SolverIterationStats))
	}
}
