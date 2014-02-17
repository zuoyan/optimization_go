package main

import (
	"fmt"
	"optimization"
)

func ipow(x float64, n int) float64 {
	r := 1.0
	for i := 0; i < n; i++ {
		r = r * x
	}
	return r
}

func square(x float64) float64 {
	return x * x
}

func opt_func(v optimization.Point) float64 {
	x, y := v.Dense[0], v.Dense[1]
	return ipow(x-1.5, 4) + ipow(y-2.5, 4)
}

func opt_grad(v optimization.Point) optimization.Point {
	x, y := v.Dense[0], v.Dense[1]
	gx, gy := 4*ipow(x-1.5, 3), 4*ipow(y-2.5, 3)
	return optimization.DensePoint([]float64{gx, gy})
}

func test_solver(name string, solver optimization.Solver) {
	p := optimization.DensePoint([]float64{2.0, 2.0})
	problem := &optimization.Problem{
		Value:    opt_func,
		Gradient: opt_grad,
	}
	v, m := solver.Solve(problem, p)
	fmt.Printf("solver %s min value %v\n", name, v)
	fmt.Printf("solver %s min at %v\n", name, m.Dense)
}

func main() {
	test_solver("gradient", &optimization.GradientDescentSolver{MaxIter: 30})
	test_solver("conjugate", &optimization.ConjugateGradientSolver{MaxIter: 30})
	test_solver("lm_bfgs", &optimization.LmBFGSSolver{MaxIter: 30})
}
