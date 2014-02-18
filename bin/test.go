package main

import (
	"flag"
	"log"
	"math/rand"
	"optimization"
)

var (
	cf       = 0
	cg       = 0
	row      = flag.Int("row", 4, "row of matrix")
	col      = flag.Int("col", 4, "col of matrix")
	max_iter = flag.Int("max_iter", 30, "max iteration")
)

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
	cf += 1
	r := mv(A, v.Dense)
	s := 0.0
	for i := 0; i < len(r); i++ {
		r[i] *= v.Factor
		r[i] -= b[i]
		s += square(r[i])
	}
	log.Printf("caled func(%s) = %f\n", v.String(), s)
	return s
}

func opt_grad(A []float64, b []float64, v optimization.Point) optimization.Point {
	cg += 1
	r := mv(A, v.Dense)
	for i := 0; i < len(r); i++ {
		r[i] *= v.Factor
		r[i] -= b[i]
	}
	gd := mtv(A, r)
	g := optimization.Point{Factor: 2.0, Dense: gd}
	log.Printf("caled grad(%s) = %s\n", v.String(), g.String())
	return g
}

func test_solver(A, b []float64, p optimization.Point, name string, solver optimization.Solver) {
	log.Printf("solver %s ...\n", name)
	cf, cg = 0, 0
	solver.Init(map[string]interface{}{"MaxIter": *max_iter})
	problem := &optimization.Problem{
		ValueFunc:    func(p optimization.Point) float64 { return opt_func(A, b, p) },
		GradientFunc: func(p optimization.Point) optimization.Point { return opt_grad(A, b, p) },
	}
	v, m := solver.Solve(problem, p)
	log.Printf("solver %s min value %v at %v #f=%v #g=%v\n", name, v, m.Dense, cf, cg)
}

func main() {
	flag.Parse()
	A := make([]float64, *row**col)
	x := make([]float64, *col)
	for i := 0; i < len(A); i++ {
		A[i] = rand.Float64()
	}
	for i := 0; i < len(x); i++ {
		x[i] = rand.Float64()
	}
	b := mv(A, x)
	log.Printf("perfect solution at %v\n", x)
	pd := make([]float64, *col)
	for i := 0; i < *col; i++ {
		pd[i] = rand.Float64()
	}
	p := optimization.DensePoint(pd)
	log.Printf("init solution at %v\n", pd)
	test_solver(A, b, p, "lm_bfgs", &optimization.LmBFGSSolver{})
	test_solver(A, b, p, "gradient", &optimization.GradientDescentSolver{})
	test_solver(A, b, p, "conjugate", &optimization.ConjugateGradientSolver{})
}
