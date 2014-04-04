package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"optimization"
	"os"
	"strings"
)

var (
	train_file            = flag.String("train_file", "", "input real label, weak labels from file as train data set, every line should be real label, weak_0 labels, weak_1 label, ... of an sample")
	test_file             = flag.String("test_file", "", "input labels as test data set, the same format as train file")
	generate_samples      = flag.Int("generate_samples", 0, "# of samples to generate")
	generate_dimension    = flag.Int("generate_dimension", 10, "dimension of sample point")
	generate_output       = flag.String("generate_output", "", "generate output file")
	linear_eval           = flag.String("linear_eval", "", "linear weight to eval")
	calculate_weights     = flag.Bool("calculate_weights", false, "calculate weights, with assumtions that weak leaners correctnesses are normal independent")
	optimize_weights      = flag.Bool("optimize_weights", false, "optimize weights")
	optimize_weights_init = flag.String("optimize_weights_init", "", "optimize weights init values, can be cal, rand, or values")
	eval_weakers          = flag.Bool("eval_weakers", false, "evalulate weak learners")
	seed                  = flag.Int64("seed", 1, "math.rand seed")
	max_iter              = flag.Int("max_iter", 30, "max iteration of optimization")
	regular2              = flag.Float64("regular2", 1.e-4, "square regular")
	lost                  = flag.String("lost", "exp", "lost function, exp, logistic")
)

func string_to_fs(s string) []float64 {
	in := strings.NewReader(s)
	var fs []float64
	for {
		f := 0.0
		_, err := fmt.Fscan(in, &f)
		if err != nil {
			break
		}
		fs = append(fs, f)
	}
	return fs
}

func fs_to_string(fs []float64) string {
	s := ""
	for j, v := range fs {
		if j > 0 {
			s += " " + fmt.Sprintf("%v", v)
		} else {
			s += fmt.Sprintf("%v", v)
		}
	}
	return s
}

func GenerateRandom(d, n int, output_file string) {
	file, e := os.Create(output_file)
	if e != nil {
		log.Fatalf("create %v failed: %v", output_file, e)
	}
	defer file.Close()
	x := make([]float64, d)
	l := make([]float64, d+2)
	for i := 0; i < n; i++ {
		s := 0.0
		for j, _ := range x {
			x[j] = rand.Float64()
			s += x[j]
		}
		// real label, positive if average > 0.5
		if s > 0.5*float64(d) {
			l[0] = 1.0
		} else {
			l[0] = -1.0
		}
		// weak label, positive if x[j] > 0.7
		for j, v := range x {
			if v > 0.7 {
				l[j+1] = 1.0
			} else {
				l[j+1] = -1.0
			}
		}
		// weak label, positive if average > 0.45
		if s > 0.45*float64(d) {
			l[d+1] = 1.0
		} else {
			l[d+1] = -1.0
		}
		file.Write([]byte(fs_to_string(l) + "\n"))
	}
}

func LoadLabels(input_file string) [][]float64 {
	var all_labels [][]float64
	file, e := os.Open(input_file)
	if e != nil {
		log.Fatalf("open input file %v failed:%v", input_file, e)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	W := -1
	for scanner.Scan() {
		line := scanner.Text()
		labels := string_to_fs(line)
		if W == -1 {
			W = len(labels)
		}
		if len(labels) != W {
			log.Fatalf("got %v, incompatible with previous %v", len(labels), W)
		}
		all_labels = append(all_labels, labels)
	}
	if e := scanner.Err(); e != nil {
		log.Fatalf("scanner got error: %v", e)
	}
	return all_labels
}

func Precision(i int, all_labels [][]float64) float64 {
	if i == 0 {
		return 1.0
	}
	p := 0.0
	for _, labels := range all_labels {
		if labels[0]*labels[i] > 0 {
			p += 1.0
		}
	}
	return p / float64(len(all_labels))
}

func LinearPrecision(weights []float64, all_labels [][]float64) float64 {
	p := 0.0
	for _, labels := range all_labels {
		if len(labels) != len(weights)+1 {
			log.Fatalf("# label(%v) != # weight(%v) + 1", len(labels), len(weights))
		}
		s := 0.0
		for j, w := range weights {
			s += w * labels[j+1]
		}
		if s*labels[0] > 0 {
			p += 1.0
		}
	}
	return p / float64(len(all_labels))
}

func normalize(vs []float64) {
	s := 0.0
	for _, v := range vs {
		s += v
	}
	for j, _ := range vs {
		vs[j] /= s
	}
}

func CalculateWeights(all_labels [][]float64) []float64 {
	weights := make([]float64, len(all_labels[0])-1)
	for j, _ := range weights {
		p := Precision(j+1, all_labels)
		mu := p*1.0 + (1.0-p)*-1.0
		ss := p*(1.0-mu)*(1.0-mu) + (1.0-p)*(-1.0-mu)*(-1.0-mu)
		weights[j] = mu / ss
	}
	normalize(weights)
	return weights
}

func opt_func_grad(p optimization.Point, all_labels [][]float64) (float64, optimization.Point) {
	weights := optimization.VectorToDense(p)
	grads := make([]float64, len(weights))
	f := 0.0
	if *lost == "exp" {
		// e(-y)
		for _, labels := range all_labels {
			if len(labels) != len(weights)+1 {
				log.Fatalf("# label(%v) != # weight(%v) + 1", len(labels), len(weights))
			}
			s := 0.0
			for j, w := range weights {
				s += w * labels[j+1]
			}
			y := s * labels[0]
			var emy float64
			if y >= -100 {
				emy = math.Exp(-y)
			} else {
				emy = math.Exp(100)
			}
			f += emy
			for j, _ := range weights {
				grads[j] += -emy * labels[0] * labels[j+1]
			}
		}
	} else {
		for _, labels := range all_labels {
			if len(labels) != len(weights)+1 {
				log.Fatalf("# label(%v) != # weight(%v) + 1", len(labels), len(weights))
			}
			s := 0.0
			for j, w := range weights {
				s += w * labels[j+1]
			}
			y := s * labels[0]
			// log(1 + e(-y))
			if y > 0 {
				emy := math.Exp(-y)
				f += math.Log(1 + emy)
				for j, _ := range weights {
					grads[j] += -labels[0] * emy / (1 + emy) * labels[j+1]
				}
			} else {
				ey := math.Exp(y)
				f += -y + math.Log(1+ey)
				for j, _ := range weights {
					grads[j] += -labels[0] / (ey + 1) * labels[j+1]
				}
			}
		}
	}
	for j, _ := range grads {
		grads[j] /= float64(len(all_labels))
	}
	f /= float64(len(all_labels))
	if *regular2 != 0.0 {
		s := 0.0
		for j, w := range weights {
			s += w * w
			grads[j] += 2 * *regular2 * w
		}
		f += *regular2 * s
	}
	return f, optimization.VectorDensePoint(grads)
}

func OptimizeWeights(init_weights []float64, all_labels [][]float64) []float64 {
	log.Printf("optimize ...\n")
	solver := optimization.LmBFGSSolver{}
	solver.Init(map[string]interface{}{
		"MaxIter": *max_iter,
		"LogFunc": func(level int, message string) {
			log.Printf("solver[level=%v]:%v", level, message)
		},
		"MaxLogLevel": 100000,
	})
	problem := &optimization.Problem{
		ValueAndGradientFunc: func(p optimization.Point) (float64, optimization.Point) { return opt_func_grad(p, all_labels) },
	}
	m, v := solver.Solve(problem, optimization.VectorDensePoint(init_weights))
	log.Printf("solver min value %v #f=%v #g=%v at %v\n", v, problem.NumValue, problem.NumGradient, m.String())
	weights := optimization.VectorToDense(m)
	normalize(weights)
	return weights
}

func main() {
	flag.Parse()
	rand.Seed(*seed)
	if *generate_samples > 0 {
		GenerateRandom(*generate_dimension, *generate_samples, *generate_output)
		return
	}
	var train_labels [][]float64
	var test_labels [][]float64
	if len(*train_file) > 0 {
		train_labels = LoadLabels(*train_file)
	}
	if len(*test_file) > 0 {
		test_labels = LoadLabels(*test_file)
	}
	if len(*linear_eval) > 0 {
		weights := string_to_fs(*linear_eval)
		if len(train_labels) > 0 {
			p := LinearPrecision(weights, train_labels)
			fmt.Printf("at train %v\n", p)
		}
		if len(test_labels) > 0 {
			p := LinearPrecision(weights, test_labels)
			fmt.Printf("at test %v\n", p)
		}
		return
	}
	W := -1
	if len(train_labels) > 0 {
		W = len(train_labels[0]) - 1
	}
	if len(test_labels) > 0 {
		if W == -1 {
			W = len(test_labels[0]) - 1
		}
		if W != len(test_labels[0])-1 {
			log.Fatalf("incompatible test and train dataset?")
		}
	}
	if *eval_weakers {
		wps_train := make([]float64, W)
		wps_test := make([]float64, W)
		wps_all := make([]float64, W)
		if len(train_labels) > 0 {
			for j, _ := range wps_train {
				wps_train[j] = Precision(j+1, train_labels)
			}
			fmt.Printf("weak precisions at train %v\n", fs_to_string(wps_train))
		}
		if len(test_labels) > 0 {
			for j, _ := range wps_test {
				wps_test[j] = Precision(j+1, test_labels)
			}
			fmt.Printf("weak precisions at test %v\n", fs_to_string(wps_test))
		}
		if len(train_labels) > 0 && len(test_labels) > 0 {
			for j, _ := range wps_test {
				wps_all[j] = wps_train[j]*float64(len(train_labels)) + wps_test[j]*float64(len(test_labels))
				wps_all[j] /= float64(len(train_labels) + len(test_labels))
			}
			fmt.Printf("weak precisions at all %v\n", fs_to_string(wps_all))
		}
	}
	if *calculate_weights {
		weights := CalculateWeights(train_labels)
		fmt.Printf("calculate weights %v\n", fs_to_string(weights))
		var precision_train, precision_test, precision_all float64
		if len(train_labels) > 0 {
			precision_train = LinearPrecision(weights, train_labels)
			fmt.Printf("weights precision at train %v\n", precision_train)
		}
		if len(test_labels) > 0 {
			precision_test = LinearPrecision(weights, test_labels)
			fmt.Printf("weights precision at test %v\n", precision_test)
		}
		if len(train_labels) > 0 && len(test_labels) > 0 {
			precision_all = precision_train*float64(len(train_labels)) + precision_test*float64(len(test_labels))
			precision_all /= float64(len(train_labels) + len(test_labels))
			fmt.Printf("weights precision at all %v\n", precision_all)
		}
	}
	if *optimize_weights {
		var weights []float64
		if *optimize_weights_init == "cal" {
			weights = CalculateWeights(train_labels)
		} else if *optimize_weights_init == "rand" || *optimize_weights_init == "" {
			weights = make([]float64, len(train_labels[0])-1)
			for j, _ := range weights {
				weights[j] = rand.Float64()
			}
		} else {
			weights = string_to_fs(*optimize_weights_init)
		}
		normalize(weights)
		weights = OptimizeWeights(weights, train_labels)
		fmt.Printf("optimize weights %v\n", weights)
		var precision_all, precision_train, precision_test float64
		if len(train_labels) > 0 {
			precision_train = LinearPrecision(weights, train_labels)
			fmt.Printf("weights precision at train %v\n", precision_train)
		}
		if len(test_labels) > 0 {
			precision_test = LinearPrecision(weights, test_labels)
			fmt.Printf("weights precision at test %v\n", precision_test)
		}
		if len(train_labels) > 0 && len(test_labels) > 0 {
			precision_all = precision_train*float64(len(train_labels)) + precision_test*float64(len(test_labels))
			precision_all /= float64(len(train_labels) + len(test_labels))
			fmt.Printf("weights precision at all %v\n", precision_all)
		}
	}
}
