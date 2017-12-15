#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Register {
    #[cfg(test)]
    A,
    #[cfg(test)]
    B,
    #[cfg(test)]
    C,
    #[cfg(test)]
    D,
    #[cfg(test)]
    E,
    #[cfg(test)]
    H,
    #[cfg(test)]
    L,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RegisterPair {
    #[cfg(test)]
    Bc,
}
