// constrained optimization through projection
package optimization

import (
	"container/list"
	"fmt"
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

type SolverIterationStats struct {
	Iteration      int
	NumFunction    int
	NumGradient    int
	NumFunctionAll int
	NumGradientAll int
	X              Point
	Y              float64
	G              Point
	CheckResult    int
}

type Solver interface {
	Check(*Problem, SolverIterationStats, Point, float64) int
	Solve(*Problem, Point) (Point, float64)
	Init(map[string]interface{})
	Log(int, string)
	Logf(int, string, ...interface{})
	LogResult(int, func() string)
	LogIterationStats(SolverIterationStats)
}

type SolverBase struct {
	CheckFunc             func(*Problem, SolverIterationStats, Point, float64) int
	MaxIter               int
	MaxLogLevel           int
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
	if level <= solver.MaxLogLevel {
		if solver.LogFunc != nil {
			solver.LogFunc(level, message)
		}
	}
	return
}

func (solver *SolverBase) LogResult(level int, f func() string) {
	if level <= solver.MaxLogLevel {
		solver.Log(level, f())
	}
}

func (solver *SolverBase) Logf(level int, format string, vs ...interface{}) {
	solver.LogResult(level, func() string {
		return fmt.Sprintf(format, vs...)
	})
}

func (solver *SolverBase) LogIterationStats(stats SolverIterationStats) {
	if solver.LogIterationStatsFunc != nil {
		solver.LogIterationStatsFunc(stats)
	} else {
		solver.LogResult(100,
			func() string {
				return fmt.Sprintf("iter=%v y=%v #f=%v/%v #g=%v/%v result=%s",
					stats.Iteration, stats.Y, stats.NumFunction, stats.NumFunctionAll, stats.NumGradient, stats.NumGradientAll,
					CheckResultString(stats.CheckResult))
			})
	}
	return
}

func (solver *SolverBase) Init(kwds map[string]interface{}) {
	if v, ok := kwds["MaxIter"]; ok {
		solver.MaxIter = v.(int)
	} else {
		solver.MaxIter = 30
	}
	if v, ok := kwds["MaxLogLevel"]; ok {
		solver.MaxLogLevel = v.(int)
	} else {
		solver.MaxLogLevel = 1000
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
