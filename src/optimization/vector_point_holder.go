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

type VectorPointHolder struct {
	Dense  []float64
	Sparse []FeatureValue
}

func (a VectorPointHolder) Size() int {
	if a.Dense != nil {
		return len(a.Dense)
	}
	if a.Sparse != nil {
		return len(a.Sparse)
	}
	return 0
}

func (a VectorPointHolder) Equal(x float64, y float64, bphi PointHolderInterface) bool {
	b := bphi.(VectorPointHolder)
	as := a.Size()
	bs := b.Size()
	if a.Dense != nil && b.Dense != nil {
		if as != bs {
			return false
		}
		for i := 0; i < as; i++ {
			av := x * a.Dense[i]
			bv := y * b.Dense[i]
			if av != bv {
				return false
			}
		}
		return true
	}
	if a.Dense != nil {
		bi := 0
		ax := -1
		for ; bi < bs && ax < as; bi++ {
			bx := b.Sparse[bi].Feature
			for c := ax + 1; c < bx; c++ {
				if c < as {
					if x*a.Dense[c] != 0 {
						return false
					}
				}
			}
			ax = bx
			av := 0.0
			if ax < as {
				av = x * a.Dense[ax]
			}
			bv := y * b.Sparse[bi].Value
			if av != bv {
				return false
			}
		}
		for c := ax + 1; c < as; c++ {
			av := x * a.Dense[c]
			if av != 0.0 {
				return false
			}
		}
		for ; bi < bs; bi++ {
			bv := y * b.Sparse[bi].Value
			if bv != 0.0 {
				return false
			}
		}
		return true
	}
	if b.Dense != nil {
		return b.Equal(y, x, a)
	}
	ai, bi := 0, 0
	for ai < as && bi < bs {
		ax := a.Sparse[ai].Feature
		bx := b.Sparse[bi].Feature
		av, bv := 0.0, 0.0
		if ax <= bx {
			av = x * a.Sparse[ai].Value
			ai++
		}
		if bx <= ax {
			bv = y * b.Sparse[bi].Value
			bi++
		}
		if av != bv {
			return false
		}
	}
	for ; ai < as; ai++ {
		av := x * a.Sparse[ai].Value
		if av != 0.0 {
			return false
		}
	}
	for ; bi < bi; bi++ {
		bv := y * b.Sparse[bi].Value
		if bv != 0 {
			return false
		}
	}
	return true
}

func (a VectorPointHolder) String(x float64) string {
	s := a.Size()
	if a.Dense != nil {
		ret := "["
		for i, v := range a.Dense {
			if i > 0 {
				ret += ", "
			}
			ret += fmt.Sprintf("%v", v*x)
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
		ret += fmt.Sprintf("%v:%v", fv.Feature, x*fv.Value)
	}
	return ret + "}"
}

func (a VectorPointHolder) AbsMax(x float64) float64 {
	if x == 0 || a.Size() == 0 {
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
	return m * abs(x)
}

func (a VectorPointHolder) SquareSum(x float64) float64 {
	if x == 0 || a.Size() == 0 {
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
	return m * x * x
}

type fsic struct {
	f float64
	s []FeatureValue
	i int
	c int
}

type fsic_heap struct {
	vs []fsic
	n  int
}

func (a *fsic_heap) Len() int {
	return a.n
}

func (a *fsic_heap) Less(i, j int) bool {
	return a.vs[i].c < a.vs[j].c
}

func (a *fsic_heap) Swap(i, j int) {
	a.vs[i], a.vs[j] = a.vs[j], a.vs[i]
}

func (a *fsic_heap) Push(x interface{}) {
	if len(a.vs) <= a.n {
		log.Fatalf("push overflow n=%v", a.n)
	}
	a.vs[a.n] = x.(fsic)
	a.n++
}

func (a *fsic_heap) Pop() interface{} {
	if a.n == 0 {
		log.Fatalf("underflow")
	}
	a.n--
	return a.vs[a.n]
}

func plusSparse(df *float64, dh *VectorPointHolder, fs []float64, ss [][]FeatureValue) {
	if len(ss) == 0 {
		return
	}
	if len(fs) != len(ss) {
		log.Fatalf("local point plus incomptaible size")
	}
	for i := 0; i < len(fs); i++ {
		if len(ss[i]) == 0 {
			log.Fatalf("local point plus with zero Size(%v)", len(ss[i]))
		}
		if fs[i] == 0.0 {
			log.Fatalf("local point plus with zero factor")
		}
	}
	if dh.Size() == 0 {
		*df = 0.0
	}
	c := 0
	if *df == 0.0 {
		*dh = VectorPointHolder{Sparse: ss[0]}
		*df = fs[0]
		c = 1
	}
	if len(fs) == c {
		return
	}
	if dh.Dense != nil {
		// TODO: I have to copy the whole dense array ...
		d := make([]float64, dh.Size())
		for x, v := range dh.Dense {
			d[x] = v * *df
		}
		*dh = VectorPointHolder{Dense: d}
		*df = 1.0
		for i := 0; i < len(fs); i++ {
			for _, fv := range ss[i] {
				v := fv.Value * fs[i]
				x := fv.Feature
				if x >= len(dh.Dense) {
					log.Fatal("dense(%v) += sparse with overflow(%v)",
						len(dh.Dense), x)
				}
				dh.Dense[x] += v
			}
		}
		return
	}
	d := 0
	if *df != 0.0 {
		d = 1
	}
	cans := &fsic_heap{
		n:  0,
		vs: make([]fsic, len(ss)-c+d),
	}
	nz := 0
	last_c := -1
	for i := c; i < len(ss); i++ {
		e := fsic{f: fs[i], s: ss[i], i: 0, c: ss[i][0].Feature}
		heap.Push(cans, e)
	}
	if d == 1 {
		e := fsic{f: *df, s: dh.Sparse, i: 0, c: dh.Sparse[0].Feature}
		heap.Push(cans, e)
	}
	for cans.Len() > 0 {
		e := &cans.vs[0]
		if last_c != e.c && e.s[e.i].Value != 0.0 {
			nz++
			last_c = e.c
		}
		if e.i+1 < len(e.s) {
			e.i++
			e.c = e.s[e.i].Feature
			heap.Fix(cans, 0)
		} else {
			heap.Remove(cans, 0)
		}
	}
	for i := c; i < len(fs); i++ {
		e := fsic{f: fs[i], s: ss[i], i: 0, c: ss[i][0].Feature}
		heap.Push(cans, e)
	}
	if d == 1 {
		e := fsic{f: *df, s: dh.Sparse, i: 0, c: dh.Sparse[0].Feature}
		heap.Push(cans, e)
	}
	s := make([]FeatureValue, nz)
	last_c = -1
	index := -1
	for cans.Len() > 0 {
		e := &cans.vs[0]
		v := e.s[e.i].Value * e.f
		if last_c != e.c && v != 0.0 {
			index++
			last_c = e.c
			s[index].Feature = e.c
		}
		s[index].Value += v
		e.i++
		if e.i < len(e.s) {
			e.c = e.s[e.i].Feature
			heap.Fix(cans, 0)
		} else {
			heap.Remove(cans, 0)
		}
	}
	*dh = VectorPointHolder{Sparse: s[:index+1]}
	*df = 1.0
	return
}

func plusDense(df *float64, dh *VectorPointHolder, fs []float64, ds [][]float64) {
	if len(ds) == 0 {
		return
	}
	if len(fs) != len(ds) {
		log.Fatalf("incompatible size")
	}
	for i := 0; i < len(fs); i++ {
		if len(ds[i]) != len(ds[0]) {
			log.Fatalf("Size(%v and %v) uncompatible to sum", len(ds[0]), len(ds[i]))
		}
		if fs[i] == 0 {
			log.Fatalf("plus with zero factor")
		}
	}
	if dh.Dense != nil && len(ds[0]) != dh.Size() {
		log.Fatalf("Size(%v and %v) uncompatible to sum", len(ds[0]), dh.Size())
	}
	if dh.Size() == 0 {
		*df = 0.0
	}
	is_alloc := false
	c := 0
	if *df == 0.0 {
		*dh = VectorPointHolder{Dense: ds[0]}
		*df = fs[0]
		is_alloc = false
		c = 1
	}
	size := len(ds[0])
	if len(ds) > c && is_alloc == false {
		t := VectorPointHolder{Dense: make([]float64, size)}
		if dh.Dense != nil {
			for x, v := range dh.Dense {
				t.Dense[x] = v * *df
			}
		} else if dh.Sparse != nil {
			for _, fv := range dh.Sparse {
				x := fv.Feature
				v := fv.Value * *df
				if x >= size {
					log.Fatalf("sparse index(%v) is larger than dense size(%v)", x, size)
				}
				t.Dense[x] += v
			}
		}
		*dh = t
		*df = 1.0
		is_alloc = true
	}
	for ; c < len(ds); c++ {
		if *df != 1.0 {
			for x, v := range dh.Dense {
				dh.Dense[x] = v * *df
			}
			*df = 1.0
		}
		for x, v := range ds[c] {
			dh.Dense[x] += fs[c] * v
		}
	}
	return
}

func remove_if(fs []float64, hs []PointHolderInterface, n int, pred func(float64, PointHolderInterface) bool) int {
	f := 0
	t := 0
	for f+t < n {
		if pred(fs[f], hs[f]) {
			t++
			hs[n-t], hs[f] = hs[f], hs[n-t]
			fs[n-t], fs[f] = fs[f], fs[n-t]
		} else {
			f++
		}
	}
	return f
}

func (h VectorPointHolder) LinearPlus(f float64, fs []float64, hs []PointHolderInterface) (float64, PointHolderInterface) {
	n := len(fs)
	if len(fs) != len(hs) {
		log.Fatalf("incomaptible size")
	}
	n = remove_if(fs, hs, n,
		func(cf float64, cphi PointHolderInterface) bool {
			ch := cphi.(VectorPointHolder)
			if cf == 0 || ch.Size() == 0 {
				return true
			}
			if ch.Dense != nil && h.Dense != nil && &ch.Dense[0] == &h.Dense[0] {
				f += cf
				return true
			}
			if ch.Sparse != nil && h.Sparse != nil && &ch.Sparse[0] == &h.Sparse[0] {
				f += cf
				return true
			}
			return false
		})
	if n == 0 {
		return f, h
	}
	if h.Size() == 0 {
		f = 0.0
	}
	nd := remove_if(fs, hs, n, func(f float64, h PointHolderInterface) bool { return h.(VectorPointHolder).Dense == nil })
	if nd > 0 {
		ds := make([][]float64, nd)
		for i := 0; i < nd; i++ {
			ds[i] = hs[i].(VectorPointHolder).Dense
		}
		plusDense(&f, &h, fs[0:nd], ds)
	}
	if nd < n {
		ss := make([][]FeatureValue, n-nd)
		for i := nd; i < n; i++ {
			ss[i-nd] = hs[i].(VectorPointHolder).Sparse
		}
		plusSparse(&f, &h, fs[nd:n], ss)
	}
	return f, h
}

func (a VectorPointHolder) InnerProd(x, y float64, bphi PointHolderInterface) float64 {
	b := bphi.(VectorPointHolder)
	if x == 0 || y == 0 || a.Size() == 0 || b.Size() == 0 {
		return 0.0
	}
	s := 0.0
	if a.Dense != nil && b.Dense != nil {
		if a.Size() != b.Size() {
			log.Fatalf("incompatible size")
		}
		for i, av := range a.Dense {
			s += av * b.Dense[i]
		}
		return s * x * y
	}
	if a.Dense != nil {
		for _, bfv := range b.Sparse {
			x := bfv.Feature
			bv := bfv.Value
			if x < len(a.Dense) {
				s += a.Dense[x] * bv
			}
		}
		return s * x * y
	}
	if b.Dense != nil {
		return b.InnerProd(y, x, a)
	}
	ai, bi := 0, 0
	as, bs := a.Size(), b.Size()
	for ai < as && bi < bs {
		ax := a.Sparse[ai].Feature
		bx := b.Sparse[bi].Feature
		if ax < bx {
			ai++
		} else if bx < ax {
			bi++
		} else {
			av := x * a.Sparse[ai].Value
			bv := y * b.Sparse[bi].Value
			ai++
			bi++
			s += av * bv
		}
	}
	return s * x * y
}

func VectorDensePoint(a []float64) Point {
	d := make([]float64, len(a))
	for x, v := range a {
		d[x] = v
	}
	return Point{Factor: 1.0, Holder: VectorPointHolder{Dense: d}}
}

func VectorToDense(a Point) []float64 {
	d := make([]float64, a.Size())
	h := a.Holder.(VectorPointHolder)
	if h.Dense != nil {
		for x, v := range h.Dense {
			d[x] = v * a.Factor
		}
	} else {
		for i, fv := range h.Sparse {
			d[i] = fv.Value * a.Factor
		}
	}
	return d
}
