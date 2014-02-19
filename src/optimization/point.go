package optimization

import (
	"container/heap"
	"fmt"
	"log"
)

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
	d := make([]float64, len(a))
	for x, v := range a {
		d[x] = v
	}
	return Point{Factor: 1.0, Dense: d}
}

func (a Point) ToDense() []float64 {
	d := make([]float64, a.Size())
	if a.IsDense() {
		for x, v := range a.Dense {
			d[x] = v * a.Factor
		}
	} else {
		for i, fv := range a.Sparse {
			d[i] = fv.Value * a.Factor
		}
	}
	return d
}

func (a Point) Size() int {
	if a.Dense != nil {
		return len(a.Dense)
	}
	if a.Sparse != nil {
		return len(a.Sparse)
	}
	return 0
}

func (a Point) IsDense() bool {
	return a.Dense != nil
}

func (a Point) Equal(b Point) bool {
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
		return b.Equal(a)
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

func (a Point) String() string {
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

func (a Point) Scale(x float64) Point {
	a.Factor *= x
	return a
}

func (a Point) AbsMax() float64 {
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

func (a Point) SquareSum() float64 {
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

func remove_if(vs []Point, n int, pred func(Point) bool) int {
	f := 0
	t := 0
	for f+t < n {
		if pred(vs[f]) {
			t++
			vs[n-t], vs[f] = vs[f], vs[n-t]
		} else {
			f++
		}
	}
	return f
}

func plusDense(dest *Point, vs []Point) {
	if len(vs) == 0 {
		return
	}
	for i := 0; i < len(vs); i++ {
		if vs[i].Size() != vs[0].Size() {
			log.Fatalf("Size(%v and %v) uncompatible to sum", vs[0].Size(), vs[1].Size())
		}
		if !vs[i].IsDense() {
			log.Fatalf("expect all dense")
		}
	}
	if dest.IsDense() && vs[0].Size() != dest.Size() {
		log.Fatalf("Size(%v and %v) uncompatible to sum", vs[0].Size(), dest.Size())
	}
	if dest.Size() == 0 {
		dest.Factor = 0.0
	}
	is_alloc := false
	c := 0
	if dest.Factor == 0.0 {
		*dest = vs[0]
		is_alloc = false
		c = 1
	}
	size := vs[0].Size()
	if len(vs) > c && is_alloc == false {
		t := Point{Factor: 1, Dense: make([]float64, size)}
		if dest.Dense != nil {
			for x, v := range dest.Dense {
				t.Dense[x] = v * dest.Factor
			}
		} else if dest.Sparse != nil {
			for _, fv := range dest.Sparse {
				x := fv.Feature
				v := fv.Value * dest.Factor
				if x >= size {
					log.Fatalf("sparse index(%v) is larger than dense size(%v)", x, size)
				}
				t.Dense[x] += v
			}
		}
		*dest = t
		is_alloc = true
	}
	for ; c < len(vs); c++ {
		if dest.Factor != 1 {
			for x, v := range dest.Dense {
				dest.Dense[x] = v * dest.Factor
			}
			dest.Factor = 1
		}
		for x, v := range vs[c].Dense {
			dest.Dense[x] += vs[c].Factor * v
		}
	}
	return
}

type point_index_col struct {
	p Point
	i int
	c int
}

type point_index_col_heap struct {
	vs []point_index_col
	n  int
}

func (a *point_index_col_heap) Len() int {
	return a.n
}

func (a *point_index_col_heap) Less(i, j int) bool {
	return a.vs[i].c < a.vs[j].c
}

func (a *point_index_col_heap) Swap(i, j int) {
	a.vs[i], a.vs[j] = a.vs[j], a.vs[i]
}

func (a *point_index_col_heap) Push(x interface{}) {
	if len(a.vs) <= a.n {
		log.Fatalf("push overflow n=%v", a.n)
	}
	a.vs[a.n] = x.(point_index_col)
	a.n++
}

func (a *point_index_col_heap) Pop() interface{} {
	if a.n == 0 {
		log.Fatalf("underflow")
	}
	a.n--
	return a.vs[a.n]
}

func plusSparse(dest *Point, vs []Point) {
	if len(vs) == 0 {
		return
	}
	for i := 0; i < len(vs); i++ {
		if vs[i].Size() == 0 {
			log.Fatalf("plus Point with zero Size(%v)", vs[i].Size())
		}
		if vs[i].IsDense() {
			log.Fatalf("expect all sparse %v/%v", i, len(vs))
		}
	}
	if dest.Size() == 0 {
		dest.Factor = 0.0
	}
	c := 0
	if dest.Factor == 0.0 {
		*dest = vs[0]
		c = 1
	}
	if len(vs) == c {
		return
	}
	if dest.IsDense() {
		// TODO: I have to copy the whole dense array ...
		d := make([]float64, dest.Size())
		for x, v := range dest.Dense {
			d[x] = v * dest.Factor
		}
		*dest = Point{Factor: 1, Dense: d}
		for i := 0; i < len(vs); i++ {
			for _, fv := range vs[i].Sparse {
				v := fv.Value * vs[i].Factor
				x := fv.Feature
				if x >= len(dest.Dense) {
					log.Fatal("dense(%v) += sparse with overflow(%v)",
						len(dest.Dense), x)
				}
				dest.Dense[x] += v
			}
		}
		return
	}
	d := 0
	if dest.Factor != 0.0 {
		d = 1
	}
	cans := &point_index_col_heap{
		n:  0,
		vs: make([]point_index_col, len(vs)-c+d),
	}
	nz := 0
	last_c := -1
	for i := c; i < len(vs); i++ {
		pic := point_index_col{p: vs[i], i: 0, c: vs[i].Sparse[0].Feature}
		heap.Push(cans, pic)
	}
	if d == 1 {
		pic := point_index_col{p: *dest, i: 0, c: dest.Sparse[0].Feature}
		heap.Push(cans, pic)
	}
	for cans.Len() > 0 {
		pic := &cans.vs[0]
		if last_c != pic.c && pic.p.Sparse[pic.i].Value != 0.0 {
			nz++
			last_c = pic.c
		}
		if pic.i+1 < pic.p.Size() {
			pic.i++
			pic.c = pic.p.Sparse[pic.i].Feature
			heap.Fix(cans, 0)
		} else {
			heap.Remove(cans, 0)
		}
	}
	for i := c; i < len(vs); i++ {
		pic := point_index_col{p: vs[i], i: 0, c: vs[i].Sparse[0].Feature}
		heap.Push(cans, pic)
	}
	if d == 1 {
		pic := point_index_col{p: *dest, i: 0, c: dest.Sparse[0].Feature}
		heap.Push(cans, pic)
	}
	s := make([]FeatureValue, nz)
	last_c = -1
	index := -1
	for cans.Len() > 0 {
		pic := &cans.vs[0]
		v := pic.p.Sparse[pic.i].Value * pic.p.Factor
		if last_c != pic.c && v != 0.0 {
			index++
			last_c = pic.c
			s[index].Feature = pic.c
		}
		s[index].Value += v
		pic.i++
		if pic.i < pic.p.Size() {
			pic.c = pic.p.Sparse[pic.i].Feature
			heap.Fix(cans, 0)
		} else {
			heap.Remove(cans, 0)
		}
	}
	*dest = Point{Factor: 0, Sparse: s[:index+1]}
	return
}

func plusImpl(dest *Point, vs []Point) {
	n := len(vs)
	n = remove_if(vs, n,
		func(a Point) bool {
			if a.Factor == 0 || a.Size() == 0 {
				return true
			}
			if a.Dense != nil && dest.Dense != nil && &a.Dense[0] == &dest.Dense[0] {
				dest.Factor += a.Factor
				return true
			}
			if a.Sparse != nil && dest.Sparse != nil && &a.Sparse[0] == &dest.Sparse[0] {
				dest.Factor += a.Factor
				return true
			}
			return false
		})
	if n == 0 {
		return
	}
	if dest.Size() == 0 {
		dest.Factor = 0.0
	}
	nd := remove_if(vs, n, func(a Point) bool { return !a.IsDense() })
	plusDense(dest, vs[0:nd])
	plusSparse(dest, vs[nd:n])
	return
}

func Sum(vs ...Point) (ret Point) {
	plusImpl(&ret, vs)
	return ret
}

func (a *Point) PlusAssign(vs ...Point) {
	plusImpl(a, vs)
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
