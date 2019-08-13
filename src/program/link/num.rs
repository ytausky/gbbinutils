use std::ops::{Add, AddAssign, BitOr, Div, Mul, RangeInclusive, Sub};

#[derive(Clone, Debug, PartialEq)]
pub enum Num {
    Range { min: i32, max: i32 },
    Unknown,
}

impl Num {
    pub fn exact(&self) -> Option<i32> {
        match *self {
            Num::Range { min, max } if min == max => Some(min),
            _ => None,
        }
    }
}

impl Default for Num {
    fn default() -> Self {
        Num::Unknown
    }
}

impl From<i32> for Num {
    fn from(n: i32) -> Self {
        Num::Range { min: n, max: n }
    }
}

impl From<RangeInclusive<i32>> for Num {
    fn from(range: RangeInclusive<i32>) -> Self {
        Num::Range {
            min: *range.start(),
            max: *range.end(),
        }
    }
}

impl AddAssign<&Num> for Num {
    fn add_assign(&mut self, rhs: &Num) {
        match (self, rhs) {
            (
                Num::Range { min, max },
                Num::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => {
                *min += rhs_min;
                *max += rhs_max;
            }
            (this, _) => *this = Num::Unknown,
        }
    }
}

impl Add for &Num {
    type Output = Num;
    fn add(self, rhs: &Num) -> Self::Output {
        let mut result = self.clone();
        result += rhs;
        result
    }
}

impl Sub for &Num {
    type Output = Num;
    fn sub(self, rhs: &Num) -> Self::Output {
        match (self, rhs) {
            (
                Num::Range { min, max },
                Num::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) => Num::Range {
                min: min - rhs_max,
                max: max - rhs_min,
            },
            _ => Num::Unknown,
        }
    }
}

impl Mul for &Num {
    type Output = Num;

    fn mul(self, rhs: &Num) -> Self::Output {
        match (self, rhs) {
            (
                Num::Range {
                    min: lhs_min,
                    max: lhs_max,
                },
                Num::Range {
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
                Num::Range {
                    min: products.clone().min().unwrap(),
                    max: products.max().unwrap(),
                }
            }
            _ => Num::Unknown,
        }
    }
}

impl BitOr for &Num {
    type Output = Num;

    fn bitor(self, rhs: &Num) -> Self::Output {
        match (self, rhs) {
            (
                Num::Range { min, max },
                Num::Range {
                    min: rhs_min,
                    max: rhs_max,
                },
            ) if min == max && rhs_min == rhs_max => (min | rhs_min).into(),
            _ => Num::Unknown,
        }
    }
}

impl Div for &Num {
    type Output = Num;

    fn div(self, rhs: &Num) -> Self::Output {
        self.exact()
            .and_then(|lhs| rhs.exact().map(|rhs| lhs / rhs))
            .map_or(Num::Unknown, Into::into)
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
        let cases: &[(Num, Num, Num)] = &triples![
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
        assert_eq!(&Num::from(0x12) | &Num::from(0x34), Num::from(0x36))
    }

    #[test]
    fn div_exact_nums() {
        assert_eq!(&Num::from(72) / &Num::from(5), Num::from(14))
    }
}
