package optimization

import "math"

type DerivativeLinePoint struct {
	problem  *Problem
	point    float64
	value    *float64
	gradient *float64
}

func (a *DerivativeLinePoint) Point() float64 {
	return a.point
}

func (a *DerivativeLinePoint) Value() float64 {
	if a.value == nil {
		v := a.problem.LineValue(a.point)
		a.value = &v
	}
	return *a.value
}

func (a *DerivativeLinePoint) Gradient() float64 {
	if a.gradient == nil {
		v := a.problem.LineGradient(a.point)
		a.gradient = &v
	}
	return *a.gradient
}

func cubicMin(ax, ay, ag, bx, by, bg float64) float64 {
	dx := bx - ax
	if dx == 0 {
		return math.MaxFloat64
	}
	dy := by - ay
	Bg := dy / dx
	T := (bg - Bg) + (ag - Bg)
	Z := T - Bg
	q := Z*Z - ag*bg
	if q < 0 {
		return math.MaxFloat64
	}
	q = math.Sqrt(q)
	if dx < 0 {
		q = -q
	}
	if (Z+ag)*q < 0 {
		if abs(Z+ag-q) > dbl_epsilon {
			return ax + dx*ag/(Z+ag-q)
		}
	}
	if abs(T) < dbl_epsilon {
		return math.MaxFloat64
	}
	return ax + dx*(Z+ag+q)/(3*T)
}

func quadMin(ax, ay, ag, bx, by float64) float64 {
	if ax == bx {
		return math.MaxFloat64
	}
	Bg := (by - ay) / (bx - ax)
	gg := (Bg - ag) / (bx - ax)
	if gg < dbl_epsilon {
		return math.MaxFloat64
	}
	return ax - ag/(2*gg)
}

func interpolate(ax, ay, ag, bx, by, bg float64) float64 {
	x := cubicMin(ax, ay, ag, bx, by, bg)
	lx := ax
	rx := bx
	if lx > rx {
		lx, rx = rx, lx
	}
	cchk := .2 * (rx - lx)
	if x >= lx+cchk && x <= rx-cchk {
		return x
	}
	x = quadMin(ax, ay, ag, bx, by)
	qchk := (rx - lx) * .01
	if x >= lx+qchk && x <= rx-qchk {
		return x
	}
	return .5 * (lx + rx)
}

func cubicInterpolate(ax, ay, ag, bx, by, bg, plx, prx float64) float64 {
	x := cubicMin(ax, ay, ag, bx, by, bg)
	lx := ax
	rx := bx
	if lx > rx {
		lx, rx = rx, lx
	}
	cchk := .2 * (rx - lx)
	if x < rx && x > lx {
		if x >= lx+cchk && x <= rx-cchk {
			return x
		}
		if prx > rx && x < lx+cchk {
			return x
		}
		if plx < lx && x > rx-cchk {
			return x
		}
	}
	x = quadMin(ax, ay, ag, bx, by)
	qchk := (rx - lx) * .01
	if x < rx && x > lx {
		if x >= lx+qchk && x <= rx-qchk {
			return x
		}
		if prx > rx && x < lx+qchk {
			return x
		}
		if plx < lx && x > rx-qchk {
			return x
		}
	}
	return .5 * (lx + rx)
}

func zoom(p *Problem, c1, c2 float64, o, al, ah DerivativeLinePoint, max_iter int) DerivativeLinePoint {
	a := al
	plx := -math.MaxFloat64
	prx := math.MaxFloat64
	for iter := 0; iter < max_iter; iter++ {
		a = DerivativeLinePoint{
			problem: p,
			point: cubicInterpolate(al.Point(), al.Value(), al.Gradient(),
				ah.Point(), ah.Value(), ah.Gradient(),
				plx, prx)}
		plx = al.Point()
		prx = ah.Point()
		if plx > prx {
			plx, prx = prx, plx
		}
		if a.Point() == math.Nextafter(al.Point(), a.Point()) ||
			a.Point() == math.Nextafter(ah.Point(), a.Point()) {
			break
		}
		if a.Value() > o.Value()+c1*a.Point()*o.Gradient() || a.Value() >= al.Value() {
			ah = a
		} else {
			if abs(a.Gradient()) <= -c2*o.Gradient() {
				break
			}
			if a.Gradient()*(ah.Point()-al.Point()) >= 0 {
				ah = al
			}
			al = a
		}
	}
	if a.Value() < al.Value() && a.Value() < ah.Value() {
		return a
	}
	if al.Value() < ah.Value() {
		return al
	}
	return ah
}

func BacktrackingLineSearch(p *Problem, alpha float64) float64 {
	const sufficient_decrease = .25 // should be in (0, .5),
	const backstep = .7             // should be in (0, 1)
	const alpha_epsilon = 1.e-100

	if alpha == 0 {
		alpha = 1.0
	}
	O := DerivativeLinePoint{problem: p, point: 0.0}
	T := DerivativeLinePoint{problem: p, point: alpha}
	for T.Value() > O.Value()+sufficient_decrease*T.Point()*O.Gradient() {
		T = DerivativeLinePoint{problem: p, point: T.Point() * backstep}
		if T.Point() <= alpha_epsilon {
			break
		}
	}
	return T.Point()
}

func StrongWolfeLineSearch(p *Problem, alpha float64) float64 {
	const alpha_max = math.MaxFloat64
	const max_iter = 10
	const zoom_max_iter = 10
	const sufficient_decrease = 1.e-4
	const curvature = 0.1

	O := DerivativeLinePoint{problem: p, point: 0}
	if alpha == 0 {
		x_infty := p.LinePoint.AbsMax()
		if x_infty > 0 {
			g := p.Gradient(p.LinePoint)
			g_infty := g.AbsMax()
			alpha = .01 * x_infty / g_infty
		}
	}
	if alpha == 0 {
		alpha = 1.0
	}
	A := DerivativeLinePoint{problem: p, point: alpha}
	Ap := O
	for iter := 0; iter < max_iter; iter++ {
		if A.Value() > O.Value()+sufficient_decrease*A.Point()*O.Gradient() || (A.Value() >= Ap.Value() && iter > 0) {
			X := zoom(p, sufficient_decrease, curvature, O, Ap, A, zoom_max_iter)
			return X.Point()
		}
		if abs(A.Gradient()) <= -curvature*O.Gradient() {
			return A.Point()
		}
		if A.Gradient() >= 0 {
			X := zoom(p, sufficient_decrease, curvature, O, A, Ap, zoom_max_iter)
			return X.Point()
		}
		if A.Point() >= alpha_max {
			break
		}
		x := cubicMin(Ap.Point(), Ap.Value(), Ap.Gradient(), A.Point(), A.Value(), A.Gradient())
		Ap = A
		if x <= A.Point() || x >= alpha_max {
			x = A.Point() * 2.0 * (float64(iter) + 2.0)
		}
		A = DerivativeLinePoint{problem: p, point: x}
	}
	if A.Value() < O.Value() {
		return A.Point()
	}
	return O.Point()
}
