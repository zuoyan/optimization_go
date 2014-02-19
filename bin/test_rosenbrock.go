package main

import (
	"log"
	"optimization"
)

func square(x float64) float64 {
	return x * x
}

func opt_func(a optimization.Point) float64 {
	value := 0.0
	vs := a.Dense
	for i := 0; i < len(vs)-1; i++ {
		value += square(1.0-a.Factor*vs[i]) +
			100.0*square(a.Factor*vs[i+1]-square(a.Factor*vs[i]))
	}
	return value
}

func opt_grad(a optimization.Point) optimization.Point {
	vs := a.Dense
	gradient := make([]float64, len(vs))
	gradient[0] = -400.0*a.Factor*vs[0]*(a.Factor*vs[1]-square(a.Factor*vs[0])) -
		2.0*(1.0-a.Factor*vs[0])
	var i int
	for i = 1; i < len(vs)-1; i++ {
		gradient[i] = -400.0*a.Factor*vs[i]*(a.Factor*vs[i+1]-square(a.Factor*vs[i])) -
			2.0*(1.0-101.0*a.Factor*vs[i]+100.0*square(a.Factor*vs[i-1]))
	}
	gradient[i] = 200.0 * (a.Factor*vs[i] - square(a.Factor*vs[i-1]))
	return optimization.Point{Factor: 1, Dense: gradient}
}

func test_solver(p optimization.Point, name string, solver optimization.Solver) {
	log.Printf("solver %s ...\n", name)
	solver.Init(map[string]interface{}{
		"MaxIter": 30,
		"LogFunc": func(level int, message string) {
			log.Print(message)
		},
	})
	problem := &optimization.Problem{
		ValueFunc:    opt_func,
		GradientFunc: opt_grad,
	}
	m, y := solver.Solve(problem, p)
	log.Printf("solver %s min value %v at %v #f=%v #g=%v\n", name, y, m.Dense, problem.NumValue, problem.NumGradient)
}

func main() {
	x := []float64{11.0, 10.0}
	log.Printf("init solution at %v\n", x)
	p := optimization.DensePoint(x)
	log.Printf("perfect solution at %v\n", []float64{1.0, 1.0})
	test_solver(p, "lm_bfgs", &optimization.LmBFGSSolver{})
	test_solver(p, "gradient", &optimization.GradientDescentSolver{})
	test_solver(p, "conjugate", &optimization.ConjugateGradientSolver{})
}
