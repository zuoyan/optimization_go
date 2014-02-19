package optimization

import "fmt"

type FeatureValue struct {
	Feature int
	Value   float64
}

type Point struct {
	Factor float64
	Dense  []float64
	Sparse []FeatureValue
}

func DensePoint(a []float64) Point {
	return Point{Factor: 1.0, Dense: a}
}

func (a *Point) Size() int {
	if a.Dense != nil {
		return len(a.Dense)
	}
	if a.Sparse != nil {
		return len(a.Sparse)
	}
	return 0
}

func (a *Point) IsDense() bool {
	return a.Dense != nil
}

func (a *Point) Equal(b Point) bool {
	as := a.Size()
	bs := b.Size()
	if a.IsDense() && b.IsDense() {
		if as != bs {
			return false
		}
		for i := 0; i < as; i++ {
			av := a.Factor * a.Dense[i]
			bv := b.Factor * b.Dense[i]
			if av != bv {
				return false
			}
		}
		return true
	}
	if a.IsDense() {
		bi := 0
		ax := -1
		for ; bi < bs && ax < as; bi++ {
			bx := b.Sparse[bi].Feature
			for x := ax + 1; x < bx; x++ {
				if x < as {
					if a.Factor*a.Dense[x] != 0 {
						return false
					}
				}
			}
			ax = bx
			av := 0.0
			if ax < as {
				av = a.Factor * a.Dense[ax]
			}
			bv := b.Factor * b.Sparse[bi].Value
			if av != bv {
				return false
			}
		}
		for x := ax + 1; x < as; x++ {
			av := a.Factor * a.Dense[x]
			if av != 0.0 {
				return false
			}
		}
		for ; bi < bs; bi++ {
			bv := b.Factor * b.Sparse[bi].Value
			if bv != 0.0 {
				return false
			}
		}
		return true

	}
	if b.IsDense() {
		return b.Equal(*a)
	}
	ai, bi := 0, 0
	for ai < as && bi < bs {
		ax := a.Sparse[ai].Feature
		bx := b.Sparse[bi].Feature
		av, bv := 0.0, 0.0
		if ax <= bx {
			av = a.Factor * a.Sparse[ai].Value
			ai++
		}
		if bx <= ax {
			bv = b.Factor * b.Sparse[bi].Value
			bi++
		}
		if av != bv {
			return false
		}
	}
	for ; ai < as; ai++ {
		av := a.Factor * a.Sparse[ai].Value
		if av != 0.0 {
			return false
		}
	}
	for ; bi < bi; bi++ {
		bv := b.Factor * b.Sparse[bi].Value
		if bv != 0 {
			return false
		}
	}
	return true
}

func (a *Point) String() string {
	s := a.Size()
	if a.IsDense() {
		ret := "["
		for x, v := range a.Dense {
			if x > 0 {
				ret += ", "
			}
			ret += fmt.Sprintf("%v", v*a.Factor)
		}
		return ret + "]"
	}
	if s == 0 {
		return "{}"
	}
	ret := "{"
	for i, fv := range a.Sparse {
		if i > 0 {
			ret += ", "
		}
		ret += fmt.Sprintf("%v:%v", fv.Feature, fv.Value)
	}
	return ret + "}"
}

func (a *Point) Scale(x float64) Point {
	b := *a
	b.Factor *= x
	return b
}

func (a *Point) AbsMax() float64 {
	if a.Factor == 0 || a.Size() == 0 {
		return 0
	}
	m := 0.0
	if a.Dense != nil {
		for _, v := range a.Dense {
			if abs(v) > m {
				m = abs(v)
			}
		}
	} else if a.Sparse != nil {
		for _, fv := range a.Sparse {
			v := fv.Value
			if abs(v) > m {
				m = abs(v)
			}
		}
	}
	return m * abs(a.Factor)
}

func (a *Point) SquareSum() float64 {
	if a.Factor == 0 || a.Size() == 0 {
		return 0
	}
	m := 0.0
	if a.Dense != nil {
		for _, v := range a.Dense {
			m += v * v
		}
	} else if a.Sparse != nil {
		for _, fv := range a.Sparse {
			v := fv.Value
			m += v * v
		}
	}
	return m * a.Factor * a.Factor
}

func remove_if(ds []Point, pred func(Point) bool) int {
	n := len(ds)
	f := 0
	t := 0
	for f+t < n {
		if pred(ds[f]) {
			t++
			ds[n-t], ds[f] = ds[n-t], ds[f]
		} else {
			f++
		}
	}
	return f
}

func plusImpl(dest *Point, vs []Point) {
	if len(vs) == 0 {
		return
	}
	if dest.Size() == 0 {
		dest.Factor = 0.0
	}
	// TODO: I stop here ...
	ds := vs
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
