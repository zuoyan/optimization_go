package optimization

import (
	"fmt"
	"log"
)

type MapPointHolder struct {
	Values map[int]float64
}

func MapPoint(vs map[int]float64) Point {
	c := map[int]float64{}
	for k, v := range vs {
		c[k] = v
	}
	return Point{
		Factor: 1,
		Holder: MapPointHolder{Values: c}}
}

func (a MapPointHolder) Equal(x float64, y float64, bphi PointHolderInterface) bool {
	b := bphi.(MapPointHolder)
	for k, av := range a.Values {
		bv := b.Values[k]
		if av*x != bv*y {
			return false
		}
	}
	for k, bv := range b.Values {
		av := a.Values[k]
		if av*x != bv*y {
			return false
		}
	}
	return true
}

func (a MapPointHolder) String(x float64) string {
	ret := "{"
	for k, v := range a.Values {
		if len(ret) > 1 {
			ret += ","
		}
		ret += fmt.Sprintf("%v:%v", k, v*x)
	}
	return ret + "}"
}

func (a MapPointHolder) AbsMax() float64 {
	m := 0.0
	for _, v := range a.Values {
		if abs(v) >= m {
			m = abs(v)
		}
	}
	return m
}

func (a MapPointHolder) SquareSum() float64 {
	s := 0.0
	for _, v := range a.Values {
		s += v * v
	}
	return s
}

func (a MapPointHolder) InnerProd(bphi PointHolderInterface) float64 {
	s := 0.0
	b := bphi.(MapPointHolder)
	if len(a.Values) < len(b.Values) {
		for k, av := range a.Values {
			bv := b.Values[k]
			s += av * bv
		}
	} else {
		for k, bv := range b.Values {
			av := a.Values[k]
			s += av * bv
		}
	}
	return s

}

func (h MapPointHolder) LinearPlus(f float64, fs []float64, hs []PointHolderInterface) (float64, PointHolderInterface) {
	if len(fs) != len(hs) {
		log.Fatalf("internal error, incomptaible size")
	}
	if len(fs) == 0 {
		return f, h
	}
	vs := map[int]float64{}
	for k, v := range h.Values {
		vs[k] += v * f
	}
	for i, hi := range hs {
		for k, v := range hi.(MapPointHolder).Values {
			vs[k] += v * fs[i]
		}
	}
	return 1.0, MapPointHolder{Values: vs}
}
