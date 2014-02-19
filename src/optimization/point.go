package optimization

import "fmt"

// TODO: Sparse
type Point struct {
	Factor float64
	Dense  []float64
}

func DensePoint(a []float64) Point {
	return Point{Factor: 1.0, Dense: a}
}

func (a *Point) Size() int {
	if a.Dense != nil {
		return len(a.Dense)
	}
	return 0
}

func (a *Point) Equal(b Point) bool {
	s := a.Size()
	if s != b.Size() {
		return false
	}
	for i := 0; i < s; i++ {
		av := a.Factor * a.Dense[i]
		bv := b.Factor * b.Dense[i]
		if av != bv {
			return false
		}
	}
	return true
}

func (a *Point) String() string {
	s := a.Size()
	if s == 0 {
		return "[]"
	}
	ret := "["
	for x, v := range a.Dense {
		if x > 0 {
			ret += ", "
		}
		ret += fmt.Sprintf("%v", v*a.Factor)
	}
	return ret + "]"
}

func (a *Point) Scale(x float64) Point {
	return Point{Factor: a.Factor * x,
		Dense: a.Dense}
}

func (a *Point) AbsMax() float64 {
	if a.Factor == 0 {
		return 0
	}
	m := 0.0
	if a.Dense != nil {
		for _, v := range a.Dense {
			if abs(v) > m {
				m = abs(v)
			}
		}
	}
	return m * abs(a.Factor)
}

func (a *Point) SquareSum() float64 {
	if a.Factor == 0 {
		return 0
	}
	m := 0.0
	if a.Dense != nil {
		for _, v := range a.Dense {
			m += v * v
		}
	}
	return m * a.Factor * a.Factor
}

func plusImpl(dest *Point, ds []Point) {
	if len(ds) == 0 {
		return
	}
	if dest.Size() == 0 {
		dest.Factor = 0.0
	}
	is_alloc := false
	if len(ds) > 0 {
		c := 0
		if dest.Factor == 0.0 {
			*dest = ds[0]
			is_alloc = false
			c = 1
		}
		if len(ds) > c && is_alloc == false {
			t := Point{Factor: 1, Dense: make([]float64, len(ds[0].Dense))}
			if dest.Dense != nil {
				for x, v := range dest.Dense {
					t.Dense[x] = v * dest.Factor
				}
			}
			*dest = t
			is_alloc = true
		}
		for ; c < len(ds); c++ {
			if dest.Factor != 1 {
				for x, v := range dest.Dense {
					dest.Dense[x] = v * dest.Factor
				}
				dest.Factor = 1
			}
			for x, v := range ds[c].Dense {
				dest.Dense[x] += ds[c].Factor * v
			}
		}
	}
	return
}

func Sum(vs ...Point) (ret Point) {
	plusImpl(&ret, vs)
	return ret
}

func PlusAssign(d *Point, vs ...Point) {
	plusImpl(d, vs)
}

func (a *Point) InnerProd(b Point) float64 {
	if a.Size() == 0 || b.Size() == 0 {
		return 0
	}
	s := 0.0
	for x, v := range a.Dense {
		s += v * b.Dense[x]
	}
	return s * a.Factor * b.Factor
}
