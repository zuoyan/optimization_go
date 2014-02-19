package optimization

type LmBFGSSolver struct {
	SolverBase
	Recent int
}

func (solver *LmBFGSSolver) Init(kwds map[string]interface{}) {
	solver.SolverBase.Init(kwds)
	if v, ok := kwds["Recent"]; ok {
		solver.Recent = v.(int)
	} else {
		solver.Recent = 5
	}
}

func (solver *LmBFGSSolver) Solve(problem *Problem, x Point) (Point, float64) {
	line_search := solver.Line
	recent := solver.Recent
	max_iter := solver.MaxIter
	alpha := 0.0
	x = problem.Project(x)
	y := problem.Value(x)
	yp := y
	pre_is_rollback := false
	recent_start := 0
	d := Point{}
	g := problem.GradientProject(x, problem.Gradient(x))
	logs_rho := make([]float64, recent)
	logs_alpha := make([]float64, recent)
	logs_dg := make([]Point, recent)
	logs_dx := make([]Point, recent)
	for iter := 0; iter < max_iter; iter++ {
		solver.Logf(1000, "iter=%v start ...", iter)
		numf, numg := problem.NumValue, problem.NumGradient
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
			solver.Logf(1000, "direction from gradient descent")
			d = g.Scale(-1)
		} else {
			solver.Logf(1000, "direction from bfgs")
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
			d = problem.DirectionProject(x, d.Scale(-1))
		}
		dg := d.InnerProd(g)
		if dg >= 0 {
			solver.Logf(1000, "dg=%v > 0, reset to -g", dg)
			d = problem.DirectionProject(x, g.Scale(-1))
			dg = d.InnerProd(g)
			recent_start = iter
		}
		solver.Logf(1000, "y=%v dg=%v", y, dg)
		solver.Logf(10000, "x=%s g=%s d=%s", x.String(), g.String(), d.String())
		if dg >= 0 {
			break
		}
		problem.LineUpdate(x, d)
		// TODO: how about cauchy point here?
		if iter > 0 {
			a2 := 2 * (y - yp) / dg
			solver.Logf(1000, "a2=%v", a2)
			alpha = min(1.0, 1.01*a2)
		}
		if alpha <= 0.0 {
			alpha = 1.0
		}
		yp = y
		alpha_init := alpha
		alpha = line_search(problem, alpha_init)
		solver.Logf(1000, "line search from %v got %v", alpha_init, alpha)
		xn := problem.Project(Sum(x, d.Scale(alpha)))
		yn := problem.Value(xn)
		gn := problem.GradientProject(xn, problem.Gradient(xn))
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
		if result == BreakRollback {
			break
		}
		xp := x
		if result == BreakKeep || result == Forward {
			y = yn
			x = xn
		}
		if result < BreakMax {
			break
		}
		if iter+1 >= max_iter {
			break
		}
		if result == Forward {
			k := iter % recent
			logs_dx[k] = Sum(x, xp.Scale(-1))
			gp := g
			g = gn
			logs_dg[k] = Sum(g, gp.Scale(-1))
		} else {
			if iter == recent_start {
				break
			}
			recent_start = iter + 1
		}
		pre_is_rollback = (result == Rollback)
	}
	return x, y
}
