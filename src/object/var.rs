use std::ops::{Add, AddAssign, BitOr, Div, Mul, RangeInclusive, Sub};

#[derive(Clone, Debug, PartialEq)]
pub enum Var {
    Range { min: i32, max: i32 },
    Unknown,
}

impl Var {
    pub fn exact(&self) -> Option<i32> {
        match *self {
            Var::Range { min, max } if min == max => Some(min),
            _ => None,
        }
    }

    pub fn refine(&mut self, value: Var) -> bool {
        let old_value = self.clone();
        let was_refined = match (old_value, &value) {
            (Var::Unknown, new_value) => *new_value != Var::Unknown,
            (
                Var::Range {
                    min: old_min,
                    max: old_max,
                },
                Var::Range {
                    min: new_min,
                    max: new_max,
                },
            ) => {
                assert!(*new_min >= old_min);
                assert!(*new_max <= old_max);
                *new_min > old_min || *new_max < old_max
            }
            (Var::Range { .. }, Var::Unknown) => {
                panic!("a symbol previously approximated is now unknown")
            }
        };
        *self = value;
        was_refined
    }
}

impl Default for Var {
    fn default() -> Self {
        Var::Unknown
    }
}

impl From<i32> for Var {
    fn from(n: i32) -> Self {
        Var::Range { min: n, max: n }
    }
}

impl From<RangeInclusive<i32>> for Var {
    fn from(range: RangeInclusive<i32>) -> Self {
        Var::Range {
            min: *range.start(),
            max: *range.end(),
        }
    }
}

impl AddAssign<&Var> for Var {
    fn add_assign(&mut self, rhs: &Var) {
        match (self, rhs) {
            (
                Var::Range { min, max },
                Var::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => {
                *min += rhs_min;
                *max += rhs_max;
            }
            (this, _) => *this = Var::Unknown,
        }
    }
}

impl Add for &Var {
    type Output = Var;
    fn add(self, rhs: &Var) -> Self::Output {
        let mut result = self.clone();
        result += rhs;
        result
    }
}

impl Sub for &Var {
    type Output = Var;
    fn sub(self, rhs: &Var) -> Self::Output {
        match (self, rhs) {
            (
                Var::Range { min, max },
                Var::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => Var::Range {
                min: min - rhs_max,
                max: max - rhs_min,
            },
            _ => Var::Unknown,
        }
    }
}

impl Mul for &Var {
    type Output = Var;

    fn mul(self, rhs: &Var) -> Self::Output {
        match (self, rhs) {
            (
                Var::Range {
                    min: lhs_min,
                    max: lhs_max,
                },
                Var::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => {
                let lhs_endpoints = [lhs_min, lhs_max];
                let rhs_endpoints = [rhs_min, rhs_max];
                let products = lhs_endpoints
                    .iter()
                    .cloned()
                    .flat_map(|n| rhs_endpoints.iter().map(move |&m| m * n));
                Var::Range {
                    min: products.clone().min().unwrap(),
                    max: products.max().unwrap(),
                }
            }
            _ => Var::Unknown,
        }
    }
}

impl BitOr for &Var {
    type Output = Var;

    fn bitor(self, rhs: &Var) -> Self::Output {
        match (self, rhs) {
            (
                Var::Range { min, max },
                Var::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) if min == max && rhs_min == rhs_max => (min | rhs_min).into(),
            _ => Var::Unknown,
        }
    }
}

impl Div for &Var {
    type Output = Var;

    fn div(self, rhs: &Var) -> Self::Output {
        self.exact()
            .and_then(|lhs| rhs.exact().map(|rhs| lhs / rhs))
            .map_or(Var::Unknown, Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! triples {
        ($(($lhs:expr, $rhs:expr, $result:expr)),*) => {
            [$(($lhs.into(), $rhs.into(), $result.into())),*]
        };
    }

    #[test]
    fn multiply_ranges() {
        let cases: &[(Var, Var, Var)] = &triples![
            (2, 3, 6),
            (-1..=1, -1..=1, -1..=1),
            (1..=6, -1, -6..=-1),
            (-6..=1, -6..=1, -6..=36)
        ];
        for (lhs, rhs, product) in cases {
            assert_eq!(lhs * rhs, *product)
        }
    }

    #[test]
    fn bitwise_or_exact_nums() {
        assert_eq!(&Var::from(0x12) | &Var::from(0x34), Var::from(0x36))
    }

    #[test]
    fn div_exact_nums() {
        assert_eq!(&Var::from(72) / &Var::from(5), Var::from(14))
    }
}
