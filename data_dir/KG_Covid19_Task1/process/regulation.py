# encoding: UTF-8
import re

# 将正则表达式编译成Pattern对象
str1 = 'hel' + '*' + 'o' + '\s*' + 'w'
str2 = 'hel*o'

str3 = 'ep\s\*ith\s\*elial\s\*barrier\s\*function\s\*genes'
pattern = re.compile(str1)

# 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
match = pattern.match('hello world!')

if match:
    # 使用Match获得分组信息
    print(match.group())

### 输出 ###
# hello
