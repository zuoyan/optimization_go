package optimization

type PointHolderInterface interface {
	Equal(float64, float64, PointHolderInterface) bool
	String(float64) string
	AbsMax() float64
	SquareSum() float64
	InnerProd(PointHolderInterface) float64
	LinearPlus(float64, []float64, []PointHolderInterface) (float64, PointHolderInterface)
}

type Point struct {
	Factor float64
	Holder PointHolderInterface
}

func (a Point) Equal(b Point) bool {
	return a.Holder.Equal(a.Factor, b.Factor, b.Holder)
}

func (a Point) String() string {
	return a.Holder.String(a.Factor)
}

func (a Point) Scale(x float64) Point {
	return Point{Factor: a.Factor * x, Holder: a.Holder}
}

func (a Point) AbsMax() float64 {
	if a.Factor != 0.0 {
		return a.Holder.AbsMax() * abs(a.Factor)
	}
	return 0.0
}

func (a Point) SquareSum() float64 {
	if a.Factor != 0.0 {
		return a.Holder.SquareSum() * a.Factor * a.Factor
	}
	return 0.0
}

func Sum(vs ...Point) (ret Point) {
	remove_if := func(vs []Point, n int, pred func(Point) bool) int {
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
	n := remove_if(vs, len(vs), func(a Point) bool { return a.Factor == 0 })
	if n > 0 {
		fs := make([]float64, n-1)
		hs := make([]PointHolderInterface, n-1)
		for i := 1; i < n; i++ {
			fs[i-1] = vs[i].Factor
			hs[i-1] = vs[i].Holder
		}
		f, h := vs[0].Holder.LinearPlus(vs[0].Factor, fs, hs)
		ret = Point{Factor: f, Holder: h}
	}
	return
}

func (a *Point) PlusAssign(vs ...Point) {
	*a = Sum(append(vs, *a)...)
}

func (a *Point) InnerProd(b Point) float64 {
	if a.Factor == 0.0 || b.Factor == 0.0 {
		return 0
	}
	return a.Holder.InnerProd(b.Holder) * a.Factor * b.Factor
}
