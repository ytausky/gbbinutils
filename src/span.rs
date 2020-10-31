use crate::codebase::TextCache;
use crate::object::*;

use std::cmp::Ordering;
use std::ops::RangeInclusive;

#[derive(Clone, Debug, PartialEq)]
pub struct Spanned<T, S> {
    pub item: T,
    pub span: S,
}

pub trait WithSpan: Sized {
    fn with_span<S>(self, span: S) -> Spanned<Self, S> {
        Spanned { item: self, span }
    }
}

impl<T> WithSpan for T {}

pub(crate) trait SpanSource {
    type Span: Clone;
}

pub(crate) trait Source: SpanSource {
    fn span(&self) -> Self::Span;
}

pub(crate) trait MergeSpans<S> {
    fn merge_spans(&mut self, left: &S, right: &S) -> S;
}

pub(crate) trait StripSpan<S> {
    type Stripped;

    fn strip_span(&mut self, span: &S) -> Self::Stripped;
}

pub(crate) trait SpanSystem
where
    Self:
        SpanSource + MergeSpans<<Self as SpanSource>::Span> + StripSpan<<Self as SpanSource>::Span>,
{
    fn add_file_inclusion(
        &mut self,
        file_inclusion: FileInclusionMetadata<Self::Span>,
    ) -> SourceFileInclusionId;

    fn add_macro_def(&mut self, macro_def: MacroDefMetadata<Self::Span>) -> MacroDefId;

    fn add_macro_expansion(
        &mut self,
        macro_expansion: MacroExpansionMetadata<Self::Span>,
    ) -> MacroExpansionId;

    fn encode_span(&mut self, span: Span) -> Self::Span;
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrippedBufSpan {
    pub buf_id: SourceFileId,
    pub range: SourceFileRange,
}

impl TextCache {
    pub fn snippet(&self, stripped: &StrippedBufSpan) -> &str {
        &self.buf(stripped.buf_id).as_str()[stripped.range.clone()]
    }
}

impl PartialOrd for MacroExpansionPos {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.token != other.token {
            self.token.partial_cmp(&other.token)
        } else {
            match (&self.param_expansion, &other.param_expansion) {
                (Some(expansion), Some(other_expansion))
                    if expansion.param == other_expansion.param =>
                {
                    expansion.arg_token.partial_cmp(&other_expansion.arg_token)
                }
                (None, None) => Some(Ordering::Equal),
                _ => None,
            }
        }
    }
}

fn merge_macro_expansion_ranges(
    left: &RangeInclusive<MacroExpansionPos>,
    right: &RangeInclusive<MacroExpansionPos>,
) -> RangeInclusive<MacroExpansionPos> {
    assert!(left.start() <= right.end());
    left.start().clone()..=right.end().clone()
}

impl Default for SpanData {
    fn default() -> Self {
        Self {
            source_file_inclusions: Vec::default(),
            macro_defs: Vec::default(),
            macro_expansions: Vec::default(),
        }
    }
}

impl SpanSource for SpanData {
    type Span = Span;
}

impl MergeSpans<Span> for SpanData {
    fn merge_spans(&mut self, left: &Span, right: &Span) -> Span {
        match (left, right) {
            (
                Span::SourceFile {
                    inclusion_metadata,
                    range,
                },
                Span::SourceFile {
                    inclusion_metadata: other_inclusion_metadata,
                    range: other_range,
                },
            ) if inclusion_metadata == other_inclusion_metadata => Span::SourceFile {
                inclusion_metadata: *inclusion_metadata,
                range: range.start..other_range.end,
            },
            (
                Span::MacroExpansion { metadata, range },
                Span::MacroExpansion {
                    metadata: other_metadata,
                    range: other_range,
                },
            ) if metadata == other_metadata => Span::MacroExpansion {
                metadata: *metadata,
                range: merge_macro_expansion_ranges(range, other_range),
            },
            _ => unreachable!(),
        }
    }
}

impl StripSpan<Span> for SpanData {
    type Stripped = StrippedBufSpan;

    fn strip_span(&mut self, span: &Span) -> Self::Stripped {
        match span {
            Span::SourceFile {
                inclusion_metadata: SourceFileInclusionId(id),
                range,
            } => StrippedBufSpan {
                buf_id: self.source_file_inclusions[*id].file,
                range: range.clone(),
            },
            Span::MacroExpansion {
                metadata: MacroExpansionId(id),
                range,
            } => {
                let expansion = &self.macro_expansions[*id];
                let def = &self.macro_defs[expansion.def.0];
                let start = &def.body_spans[range.start().token];
                let end = &def.body_spans[range.end().token];
                let (buf_id, range) = match (start, end) {
                    (
                        Span::SourceFile {
                            inclusion_metadata: SourceFileInclusionId(id),
                            range,
                        },
                        Span::SourceFile {
                            inclusion_metadata: SourceFileInclusionId(other_id),
                            range: other_range,
                        },
                    ) if id == other_id => (
                        self.source_file_inclusions[*id].file,
                        range.start..other_range.end,
                    ),
                    _ => unimplemented!(),
                };
                StrippedBufSpan { buf_id, range }
            }
        }
    }
}

impl SpanSystem for SpanData {
    fn add_file_inclusion(
        &mut self,
        file_inclusion: FileInclusionMetadata<Self::Span>,
    ) -> SourceFileInclusionId {
        let id = SourceFileInclusionId(self.source_file_inclusions.len());
        self.source_file_inclusions.push(file_inclusion);
        id
    }

    fn add_macro_def(&mut self, macro_def: MacroDefMetadata<Self::Span>) -> MacroDefId {
        let id = MacroDefId(self.macro_defs.len());
        self.macro_defs.push(macro_def);
        id
    }

    fn add_macro_expansion(
        &mut self,
        macro_expansion: MacroExpansionMetadata<Self::Span>,
    ) -> MacroExpansionId {
        let id = MacroExpansionId(self.macro_expansions.len());
        self.macro_expansions.push(macro_expansion);
        id
    }

    fn encode_span(&mut self, span: Span) -> Self::Span {
        span
    }
}

#[cfg(test)]
pub mod fake {
    use super::*;

    use crate::diagnostics::mock::Merge;

    use std::marker::PhantomData;

    pub struct FakeSpanSystem<S> {
        next_source_file_inclusion_metadata_id: SourceFileInclusionId,
        next_macro_def_metadata_id: MacroDefId,
        next_macro_expansion_metadata_id: MacroExpansionId,
        _phantom_data: PhantomData<S>,
    }

    impl<S> Default for FakeSpanSystem<S> {
        fn default() -> Self {
            Self {
                next_source_file_inclusion_metadata_id: SourceFileInclusionId(0),
                next_macro_def_metadata_id: MacroDefId(0),
                next_macro_expansion_metadata_id: MacroExpansionId(0),
                _phantom_data: PhantomData,
            }
        }
    }

    impl<S: Clone> SpanSource for FakeSpanSystem<S> {
        type Span = S;
    }

    impl<S: Clone + Merge> MergeSpans<S> for FakeSpanSystem<S> {
        fn merge_spans(&mut self, left: &S, right: &S) -> S {
            S::merge(left.clone(), right.clone())
        }
    }

    impl<S: Clone> StripSpan<S> for FakeSpanSystem<S> {
        type Stripped = S;

        fn strip_span(&mut self, span: &S) -> Self::Stripped {
            span.clone()
        }
    }

    impl<S: Clone + Default + Merge> SpanSystem for FakeSpanSystem<S> {
        fn add_file_inclusion(
            &mut self,
            _: FileInclusionMetadata<Self::Span>,
        ) -> SourceFileInclusionId {
            let result = self.next_source_file_inclusion_metadata_id;
            self.next_source_file_inclusion_metadata_id = SourceFileInclusionId(result.0 + 1);
            result
        }

        fn add_macro_def(&mut self, _: MacroDefMetadata<Self::Span>) -> MacroDefId {
            let result = self.next_macro_def_metadata_id;
            self.next_macro_def_metadata_id = MacroDefId(result.0 + 1);
            result
        }

        fn add_macro_expansion(
            &mut self,
            _: MacroExpansionMetadata<Self::Span>,
        ) -> MacroExpansionId {
            let result = self.next_macro_expansion_metadata_id;
            self.next_macro_expansion_metadata_id = MacroExpansionId(result.0 + 1);
            result
        }

        fn encode_span(&mut self, _: Span) -> Self::Span {
            S::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codebase::TextCache;

    use std::ops::Range;

    #[test]
    fn extend_span() {
        let mut codebase = TextCache::new();
        let src = "left right";
        let buf_id = codebase.add_src_buf("/my/file", src);
        let mut spans = SpanData::default();
        let inclusion_metadata = spans.add_file_inclusion(FileInclusionMetadata {
            file: buf_id,
            from: None,
        });
        let left = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: 0..4,
        });
        let right = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: 5..10,
        });
        let combined = spans.merge_spans(&left, &right);
        assert_eq!(
            combined,
            Span::SourceFile {
                inclusion_metadata,
                range: 0..10
            }
        )
    }

    #[test]
    fn merge_simple_macro_expansion_positions() {
        let start_pos = MacroExpansionPos {
            token: 2,
            param_expansion: None,
        };
        let start = start_pos.clone()..=start_pos.clone();
        let end_pos = MacroExpansionPos {
            token: 7,
            param_expansion: None,
        };
        let end = end_pos.clone()..=end_pos.clone();
        assert_eq!(
            merge_macro_expansion_ranges(&start, &end),
            start_pos..=end_pos
        )
    }

    #[test]
    fn strip_buf_span() {
        let file = SourceFileId(7);
        let range = 0..1;
        let mut spans = SpanData::default();
        let inclusion_metadata =
            spans.add_file_inclusion(FileInclusionMetadata { file, from: None });
        let span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: range.clone(),
        });
        assert_eq!(
            spans.strip_span(&span),
            StrippedBufSpan {
                buf_id: file,
                range
            }
        )
    }

    #[test]
    fn strip_macro_span() {
        let buf_id = SourceFileId(1);
        let mut spans = SpanData::default();
        let inclusion_metadata = spans.add_file_inclusion(FileInclusionMetadata {
            file: buf_id,
            from: None,
        });
        let position = MacroExpansionPos {
            token: 0,
            param_expansion: None,
        };
        let macro_base = 0;
        let def_name_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: macro_name_range(macro_base),
        });
        let def_body_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: macro_body_range(macro_base),
        });
        let macro_def = spans.add_macro_def(MacroDefMetadata {
            name_span: def_name_span,
            param_spans: Box::new([]),
            body_spans: Box::new([def_body_span]),
        });
        let expansion_name_span = spans.encode_span(Span::SourceFile {
            inclusion_metadata,
            range: 40..50,
        });
        let macro_expansion = spans.add_macro_expansion(MacroExpansionMetadata {
            name_span: expansion_name_span,
            def: macro_def,
            arg_spans: Box::new([]),
        });
        let span = spans.encode_span(Span::MacroExpansion {
            metadata: macro_expansion,
            range: position.clone()..=position,
        });
        let stripped = StrippedBufSpan {
            buf_id,
            range: macro_body_range(macro_base),
        };
        assert_eq!(spans.strip_span(&span), stripped)
    }

    fn macro_name_range(base: usize) -> Range<usize> {
        base..base + 10
    }

    fn macro_body_range(base: usize) -> Range<usize> {
        base + 20..base + 30
    }
}
