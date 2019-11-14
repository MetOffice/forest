import string


def test_template_substitution():
    template = string.Template("${prefix}/leaf")
    result = template.substitute(prefix="/some/dir")
    expect = "/some/dir/leaf"
    assert expect == result


def test_string_template_given_extra_variables():
    template = string.Template("${prefix}/leaf")
    result = template.substitute(prefix="/some/dir", extra="not used")
    expect = "/some/dir/leaf"
    assert expect == result
