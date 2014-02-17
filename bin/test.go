package main

import (
	"fmt"
	"optimization"
)

var (
	cf = 0
	cg = 0
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
	cf += 1
	x, y := v.Dense[0], v.Dense[1]
	fmt.Printf("cal func(%f, %f)\n", x, y)
	return ipow(x-1.5, 4) + ipow(y-2.5, 4)
}

func opt_grad(v optimization.Point) optimization.Point {
	cg += 1
	x, y := v.Dense[0], v.Dense[1]
	fmt.Printf("cal grad(%f, %f)\n", x, y)
	gx, gy := 4*ipow(x-1.5, 3), 4*ipow(y-2.5, 3)
	return optimization.DensePoint([]float64{gx, gy})
}

func test_solver(name string, solver optimization.Solver) {
	fmt.Printf("solver %s ...\n", name)
	cf, cg = 0, 0
	solver.Init(map[string]interface{}{"MaxIter": 30})
	p := optimization.DensePoint([]float64{2.0, 2.0})
	problem := &optimization.Problem{
		ValueFunc:    opt_func,
		GradientFunc: opt_grad,
	}
	v, m := solver.Solve(problem, p)
	fmt.Printf("solver %s min value %v\n", name, v)
	fmt.Printf("solver %s min at %v\n", name, m.Dense)
	fmt.Printf("solver %s #f=%v  #g=%v\n", name, cf, cg)
}

func main() {
	test_solver("gradient", &optimization.GradientDescentSolver{})
	test_solver("conjugate", &optimization.ConjugateGradientSolver{})
	test_solver("lm_bfgs", &optimization.LmBFGSSolver{})
}
