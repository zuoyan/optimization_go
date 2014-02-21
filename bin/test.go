package main

import (
	"flag"
	"log"
	"math/rand"
	"optimization"

	lbfgsb "github.com/afbarnard/go-lbfgsb"
)

var (
	row      = flag.Int("row", 4, "row of matrix")
	col      = flag.Int("col", 4, "col of matrix")
	max_iter = flag.Int("max_iter", 30, "max iteration")
)

type LbfgsbSolver struct {
	optimization.SolverBase
}

func (solver *LbfgsbSolver) Solve(problem *optimization.Problem, x optimization.Point) (optimization.Point, float64) {
	optimizer := new(lbfgsb.Lbfgsb).SetFTolerance(1e-10).SetGTolerance(1e-10)
	point := optimization.VectorToDense(x)
	optimizer.SetLogger(func(info *lbfgsb.OptimizationIterationInformation) {
		if (info.Iteration-1)%10 == 0 {
			solver.Log(1000, info.Header())
		}
		solver.Log(1000, info.String())
	})
	objective := lbfgsb.GeneralObjectiveFunction{
		Function: func(p []float64) float64 {
			y := problem.Value(optimization.VectorDensePoint(p))
			return y
		},
		Gradient: func(p []float64) []float64 {
			g := problem.Gradient(optimization.VectorDensePoint(p))
			return optimization.VectorToDense(g)
		},
	}
	xfg, status := optimizer.Minimize(objective, point)
	stats := optimizer.OptimizationStatistics()
	log.Printf("stats: iters: %v; F evals: %v; G evals: %v", stats.Iterations, stats.FunctionEvaluations, stats.GradientEvaluations)
	log.Printf("status: %v", status)
	x = optimization.VectorDensePoint(xfg.X)
	if xfg.F != problem.Value(x) {
		log.Printf("error of value, %v != %v", xfg.F, problem.Value(x))
	}
	return x, xfg.F
}

func square(x float64) float64 {
	return x * x
}

func mv(A []float64, x []float64) []float64 {
	n := len(x)
	m := len(A) / n
	if len(A) != m*n {
		log.Fatalf("invalid size A#=%v x#=%v", len(A), len(x))
	}
	r := make([]float64, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			r[i] += A[i*n+j] * x[j]
		}
	}
	return r
}

func mtv(A []float64, x []float64) []float64 {
	n := len(x)
	m := len(A) / n
	if len(A) != m*n {
		log.Fatalf("invalid size A#=%v x#=%v", len(A), len(x))
	}
	r := make([]float64, m)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			r[i] += A[i+j*m] * x[j]
		}
	}
	return r
}

func opt_func(A []float64, b []float64, v optimization.Point) float64 {
	if len(A) != *row**col || len(b) != *row {
		log.Fatalf("invalid size row=%v col=%v len(A)=%v len(b)=%v", row, col, len(A), len(b))
	}
	r := mv(A, optimization.VectorToDense(v))
	s := 0.0
	for i := 0; i < len(r); i++ {
		r[i] *= v.Factor
		r[i] -= b[i]
		s += square(r[i])
	}
	// log.Printf("caled func(%s) = %f\n", v.String(), s)
	return s
}

func opt_grad(A []float64, b []float64, v optimization.Point) optimization.Point {
	r := mv(A, optimization.VectorToDense(v))
	for i := 0; i < len(r); i++ {
		r[i] *= v.Factor
		r[i] -= b[i]
	}
	gd := mtv(A, r)
	g := optimization.VectorDensePoint(gd).Scale(2.0)
	// log.Printf("caled grad(%s) = %s\n", v.String(), g.String())
	return g
}

func test_solver(A, b []float64, p optimization.Point, name string, solver optimization.Solver) {
	log.Printf("solver %s ...\n", name)
	solver.Init(map[string]interface{}{
		"MaxIter": *max_iter,
		"LogFunc": func(level int, message string) {
			log.Printf(message)
		},
	})
	problem := &optimization.Problem{
		ValueFunc:    func(p optimization.Point) float64 { return opt_func(A, b, p) },
		GradientFunc: func(p optimization.Point) optimization.Point { return opt_grad(A, b, p) },
	}
	m, v := solver.Solve(problem, p)
	log.Printf("solver %s min value %v #f=%v #g=%v at %v\n", name, v, problem.NumValue, problem.NumGradient, m.String())
}

func main() {
	flag.Parse()
	A := make([]float64, *row**col)
	x := make([]float64, *col)
	for i := 0; i < len(A); i++ {
		A[i] = rand.Float64() * 10
	}
	for i := 0; i < len(x); i++ {
		x[i] = rand.Float64() * 10
	}
	b := mv(A, x)
	log.Printf("perfect solution at %v", x)
	log.Printf("perfect value %v", opt_func(A, b, optimization.VectorDensePoint(x)))
	log.Printf("perfect gradient %v", opt_grad(A, b, optimization.VectorDensePoint(x)).String())

	pd := make([]float64, *col)
	for i := 0; i < *col; i++ {
		pd[i] = rand.Float64()
	}
	p := optimization.VectorDensePoint(pd)
	log.Printf("init solution at %v", pd)
	test_solver(A, b, p, "lm_bfgs", &optimization.LmBFGSSolver{})
	test_solver(A, b, p, "gradient", &optimization.GradientDescentSolver{})
	test_solver(A, b, p, "conjugate", &optimization.ConjugateGradientSolver{})
	test_solver(A, b, p, "lbfgsb", &LbfgsbSolver{})
}
