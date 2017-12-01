use ast;

fn parse_src(_src: &str) -> ast::AssemblyCommands {
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_src() {
        assert_eq!(parse_src(""), vec![]);
    }
}
