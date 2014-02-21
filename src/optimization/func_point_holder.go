package optimization

type PointHolderOperations struct {
	Equal      func(float64, interface{}, float64, interface{}) bool
	String     func(interface{}, float64) string
	AbsMax     func(interface{}) float64
	SquareSum  func(interface{}) float64
	InnerProd  func(interface{}, interface{}) float64
	LinearPlus func(float64, interface{}, []float64, []interface{}) (float64, interface{})
}

type FuncPointHolder struct {
	Operations *PointHolderOperations
	Value      interface{}
}

func (a FuncPointHolder) Equal(x, y float64, bphi PointHolderInterface) bool {
	b := bphi.(FuncPointHolder)
	return a.Operations.Equal(x, a.Value, y, b.Value)
}

func (a FuncPointHolder) String(x float64) string {
	return a.Operations.String(a.Value, x)
}

func (a FuncPointHolder) AbsMax() float64 {
	return a.Operations.AbsMax(a.Value)
}

func (a FuncPointHolder) SquareSum() float64 {
	return a.Operations.SquareSum(a.Value)
}

func (a FuncPointHolder) InnerProd(bphi PointHolderInterface) float64 {
	b := bphi.(FuncPointHolder)
	return a.Operations.InnerProd(a.Value, b.Value)
}

func (a FuncPointHolder) LinearPlus(f float64, fs []float64, hs []PointHolderInterface) (float64, PointHolderInterface) {
	vs := make([]interface{}, len(hs))
	for i, h := range hs {
		vs[i] = h
	}
	f, v := a.Operations.LinearPlus(f, a.Value, fs, vs)
	return f, FuncPointHolder{Operations: a.Operations, Value: v}
}
