package optimization

import "fmt"

type GradientDescentSolver struct {
	SolverBase
}

func (solver *GradientDescentSolver) Solve(problem *Problem, x Point) (Point, float64) {
	line_search := solver.Line
	x = problem.Project(x)
	y := problem.Value(x)
	pre_dg := 0.0
	alpha := 0.0
	max_iter := solver.MaxIter
	g := problem.GradientProject(x, problem.Gradient(x))
	for iter := 0; iter < max_iter; iter++ {
		numf, numg := problem.NumValue, problem.NumGradient
		solver.Logf(1000, "start iter=%v", iter)
		d := problem.DirectionProject(x, g.Scale(-1))
		dg := d.InnerProd(g)
		solver.Logf(1000, "y=%v dg=%v", y, dg)
		solver.LogResult(10000, func() string { return fmt.Sprintf("x=%s g=%s d=%s", x.String(), g.String(), d.String()) })
		if dg >= 0 {
			break
		}
		problem.LineUpdate(x, d)
		if iter > 0 {
			alpha *= pre_dg / dg
		}
		pre_dg = dg
		alpha_init := alpha
		alpha = line_search(problem, alpha_init)
		solver.Logf(5000, "line search from %v got %v", alpha, alpha_init)
		xn := Sum(x, d.Scale(alpha))
		xn = problem.Project(xn)
		gn := problem.GradientProject(xn, problem.Gradient(xn))
		yn := problem.LineValue(alpha)
		stats := SolverIterationStats{
			Iteration:      iter,
			NumFunction:    problem.NumValue - numf,
			NumFunctionAll: problem.NumValue,
			NumGradient:    problem.NumGradient - numg,
			NumGradientAll: problem.NumGradient,
			X:              xn,
			Y:              yn,
			G:              gn,
		}
		result := solver.Check(problem, stats, x, y)
		stats.CheckResult = result
		solver.LogIterationStats(stats)
		if result == BreakRollback || result == Rollback {
			break
		}
		x = xn
		y = yn
		g = gn
		if result < BreakMax {
			break
		}
	}
	return x, y
}
