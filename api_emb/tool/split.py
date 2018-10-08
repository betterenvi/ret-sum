import re

from ..metric.codenn import normalize


def split_text(text, lower=True):
    return normalize(text, lower=lower)


def split_csharp(code):
    from data.codenn.src.csharp.CSharpTemplate import parseCSharp
    code = parseCSharp(code)
    return code


# def split_sql(code):
#     from data.codenn.src.sql.SqlTemplate import SqlTemplate
#     query = SqlTemplate(code, regex=True)
#     typed_code = query.parseSql()
#     tokens = [re.sub('\s+', ' ', x.strip()) for x in typed_code]
#     return tokens

def split_sql(code):
    # Split SQL code directly, since I have pre-tokenized SQL code in Python2.
    return code.split()
