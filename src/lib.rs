mod ast;
mod codegen;
mod parse;

#[cfg(test)]
struct FirstPass;

#[cfg(test)]
impl FirstPass {
    #[cfg(test)]
    fn new(_src: &str) -> FirstPass {
        FirstPass {}
    }

    #[cfg(test)]
    fn sections(&self) -> SectionIterator {
        SectionIterator {}
    }
}

#[cfg(test)]
type Section = u8;

#[cfg(test)]
struct SectionIterator;

#[cfg(test)]
impl Iterator for SectionIterator {
    type Item = Section;

    fn next(&mut self) -> Option<Section> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_pass_on_empty_src() {
        let first_pass = FirstPass::new("");
        assert_eq!(first_pass.sections().next(), None);
    }
}
