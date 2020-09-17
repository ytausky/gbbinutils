pub trait GetString<StringId> {
    fn get_string<'a>(&'a self, id: &'a StringId) -> &str;
}

pub struct FakeStringInterner;

impl GetString<String> for FakeStringInterner {
    fn get_string<'a>(&'a self, id: &'a String) -> &str {
        id.as_ref()
    }
}
