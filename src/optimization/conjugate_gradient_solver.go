package optimization

import "math"

func ConjugateGradientBetaPRP(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g - pg> / <pg, pg>
	den := previous_gradient.SquareSum()
	num := gradient.SquareSum() - gradient.InnerProd(previous_gradient)
	if den > 0 && num > 0 {
		return num / den
	}
	return 0.0
}

func ConjugateGradientBetaFR(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g> / <pg, pg>
	num := gradient.SquareSum()
	den := previous_gradient.SquareSum()
	if den > 0 {
		return num / den
	}
	return 0
}

func ConjugateGradientBetaHS(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g - pg> / <d, g - pg>
	num := gradient.SquareSum() - gradient.InnerProd(previous_gradient)
	den := direction.InnerProd(gradient) - direction.InnerProd(previous_gradient)
	if den != 0 {
		return num / den
	}
	return 0
}

func ConjugateGradientBetaDY(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// <g, g> / <g - pg, d>
	num := gradient.SquareSum()
	den := gradient.InnerProd(direction) - previous_gradient.InnerProd(direction)
	if den != 0 {
		return num / den
	}
	return 0
}

func ConjugateGradientBetaDescent(alpha float64, direction, previous_gradient, point, gradient Point) float64 {
	// y = g - pg
	const eta2 = 1.e-4
	gg := gradient.SquareSum()
	gp := gradient.InnerProd(previous_gradient)
	gd := gradient.InnerProd(direction)
	pp := previous_gradient.SquareSum()
	pd := previous_gradient.InnerProd(direction)
	dd := direction.SquareSum()
	yy := gg + pp - 2*gp
	dy := gd - pd
	dg := gd
	gy := gg - gp
	b := (gy - 2*yy/dy*dg) / dy
	t := -1. / math.Sqrt(dd*min(eta2, pp))
	return max(t, b)
}

type ConjugateGradientSolver struct {
	SolverBase
	Beta func(float64, Point, Point, Point, Point) float64
}

func (solver *ConjugateGradientSolver) Init(kwds map[string]interface{}) {
	solver.SolverBase.Init(kwds)
	if v, ok := kwds["Beta"]; ok {
		solver.Beta = v.(func(float64, Point, Point, Point, Point) float64)
	} else {
		solver.Beta = ConjugateGradientBetaPRP
	}
}

func (solver *ConjugateGradientSolver) Solve(problem *Problem, x Point) (Point, float64) {
	line_search := solver.Line
	beta := solver.Beta
	x = problem.Project(x)
	y := problem.Value(x)
	g := problem.GradientProject(x, problem.Gradient(x))
	d := problem.DirectionProject(x, g.Scale(-1))
	pre_dg := 0.0
	alpha := 0.0
	max_iter := solver.MaxIter
	for iter := 0; iter < max_iter; iter++ {
		numf, numg := problem.NumValue, problem.NumGradient
		solver.Logf(1000, "iter=%v start ...", iter)
		dg := d.InnerProd(g)
		if dg >= 0 {
			solver.Logf(1000, "dg=%v > 0, set d = -g", dg)
			d = g.Scale(-1)
			d = problem.DirectionProject(x, d)
			dg = d.InnerProd(g)
		}
		solver.Logf(1000, "y=%v dg=%v", y, dg)
		solver.Logf(10000, "x=%s g=%s d=%s", x.String(), g.String(), d.String())
		if dg >= 0 {
			break
		}
		problem.LineUpdate(x, d)
		if iter > 0 {
			alpha = pre_dg / dg
		}
		pre_dg = dg
		alpha_init := alpha
		alpha = line_search(problem, alpha_init)
		solver.Logf(1000, "line search from %v got %v", alpha_init, alpha)
		xn := Sum(x, d.Scale(alpha))
		xn = problem.Project(xn)
		yn := problem.LineValue(alpha)
		gn := problem.GradientProject(xn, problem.Gradient(xn))
		stats := SolverIterationStats{
			Iteration:             iter,
			NumFunction:           problem.NumValue - numf,
			AccumulateNumFunction: problem.NumValue,
			NumGradient:           problem.NumGradient - numg,
			AccumulateNumGradient: problem.NumGradient,
			X: xn,
			Y: yn,
			G: gn,
		}
		result := solver.Check(problem, stats, x, y)
		stats.CheckResult = result
		solver.LogIterationStats(stats)
		if result == BreakRollback {
			break
		}
		if result == Forward || result == BreakKeep {
			x = xn
			y = yn
		}
		if iter+1 >= max_iter || result < BreakMax {
			break
		}
		if result == Forward {
			gp := g
			g = gn
			// Note: line search is not exact, so disable powell's
			// restart criteria.
			if false && abs(g.InnerProd(gp)) >= .2*g.SquareSum() {
				d = g.Scale(-1)
			} else {
				b := beta(alpha, d, gp, x, g)
				d = Sum(d.Scale(b), g.Scale(-1))
			}
		} else {
			dn := g.Scale(-1)
			if dn.Equal(d) {
				break
			}
			// restart
			d = dn
		}
	}
	return x, y
}
