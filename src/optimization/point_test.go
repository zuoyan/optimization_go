package optimization

import (
	"log"
	"testing"
)

func TestDense(t *testing.T) {
	v := []float64{1.0, 2.0, 3.0, 4.0}
	p := VectorDensePoint(v)
	v[0] = 3.14
	if p.String() != "[1, 2, 3, 4]" {
		t.Error("DensePoint doesn't copy")
	}
	q := p.Scale(2)
	if p.Factor != 1.0 {
		t.Error("Scale changed")
	}
	if q.String() != "[2, 4, 6, 8]" {
		t.Error("Scale error")
	}
}

func vi(i ...int) []int {
	return i
}

func vf(f ...float64) []float64 {
	return f
}

func make_sparse(vi []int, vf []float64) Point {
	if len(vi) != len(vf) {
		log.Fatalf("invalid arg ...")
	}
	s := make([]FeatureValue, len(vi))
	for i, f := range vf {
		x := vi[i]
		s[i].Feature = x
		s[i].Value = f
	}
	return Point{Factor: 1, Holder: VectorPointHolder{Sparse: s}}
}

func TestSparse(t *testing.T) {
	v := make_sparse(vi(1, 3), vf(1.0, 9.0))
	if v.Size() != 2 {
		t.Error("Sparse size")
	}
}

func TestSquareSum(t *testing.T) {
	v := VectorDensePoint([]float64{1.0, 2.0, 3.0, 4.0})
	if v.SquareSum() != 1.0+4.0+9.0+16.0 {
		t.Error("Square sum of dense")
	}
	v = make_sparse(vi(1, 3), vf(1.0, 9.0))
	if v.SquareSum() != 1.0+81.0 {
		t.Error("Square sum of sparse")
	}
}

func TestSumDense(t *testing.T) {
	a := VectorDensePoint([]float64{1.0, 2.0, 3.0, 4.0})
	b := a.Scale(2)
	if a.Factor != 1.0 {
		t.Errorf("scale changed left")
	}
	c := VectorDensePoint([]float64{100.0, 200.0, 300.0, 400.0})
	{
		d := Sum(a, b)
		e := "[3, 6, 9, 12]"
		if d.String() != e {
			t.Errorf("got %v expect %v", d.String(), e)
		}
	}
	{
		d := Sum(a.Scale(3), b)
		e := "[5, 10, 15, 20]"
		if d.String() != e {
			t.Errorf("got %v, expect %v", d.String(), e)
		}
		d = Sum(a.Scale(3), b.Scale(-2))
		e = "[-1, -2, -3, -4]"
		if d.String() != e {
			t.Errorf("got %v, expect %v", d.String(), e)
		}
	}
	{
		d := Sum(a, b, c)
		e := "[103, 206, 309, 412]"
		if d.String() != e {
			t.Errorf("got %v, expect %v", d.String(), e)
		}
	}
}

func TestSumSparse(t *testing.T) {
	sa := make_sparse(vi(1, 2, 3), vf(1, 4, 9))
	sb := make_sparse(vi(2, 3), vf(40, 90))
	sc := make_sparse(vi(0, 3), vf(10000, 900))
	{
		d := Sum(sa)
		log.Printf("test sum of one sparse %v ...", d.String())
		e := "{1:1, 2:4, 3:9}"
		if d.String() != e {
			t.Errorf("got %v, expect %v", d.String(), e)
		}
	}
	{
		d := Sum(sa, sb)
		log.Printf("test sum of two sparse %v ...", d.String())
		e := "{1:1, 2:44, 3:99}"
		if d.String() != e {
			t.Errorf("got %v, expect %v", d.String(), e)
		}
	}
	{
		d := Sum(sa.Scale(3), sb.Scale(2), sc.Scale(2))
		log.Printf("test sum of three sparse %v ...", d.String())
		e := "{0:20000, 1:3, 2:92, 3:2007}"
		if d.String() != e {
			t.Errorf("got %v, expect %v", d.String(), e)
		}
	}
}

func TestSumDenseSparse(t *testing.T) {
	sa := make_sparse(vi(1, 2, 3), vf(1, 4, 9))
	sb := make_sparse(vi(2, 3), vf(40, 90))
	sc := make_sparse(vi(0, 3), vf(10000, 900))
	da := VectorDensePoint([]float64{1.0, 2.0, 3.0, 4.0})
	db := VectorDensePoint([]float64{3.0, 4.0, 2.0, 1.0})
	dc := VectorDensePoint([]float64{100.0, 200.0, 300.0, 400.0})
	{
		v := Sum(sa, da)
		log.Printf("sum of sparse + dense %v ...", v.String())
		e := "[1, 3, 7, 13]"
		if v.String() != e {
			t.Errorf("got %v, expect %v", v.String(), e)
		}
	}
	{
		v := Sum(sa, da, sb, db, dc, sc)
		log.Printf("sum of [sparse] + [dense] %v ...", v.String())
		e := "[10104, 207, 349, 1404]"
		if v.String() != e {
			t.Errorf("got %v, expect %v", v.String(), e)
		}
	}
}
