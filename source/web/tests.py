
#!/usr/bin/python
# -*- coding: utf-8 -*-
#__author__ : stray_camel
#pip_source : https://mirrors.aliyun.com/pypi/simple
import sys,os
import pandas as pd
import re
line = 'Cool Canada (Collections» Canada) What it&#039;s like to own an Apple// product - The Oatmeal'
a = re.compile(r'[?!()-@#$.|» :\d\s&#;/]')
line = re.sub(a, ' ', line)
print('需要匹配的字符串：',line)
l = re.split(' +', line)
bag = pd.DataFrame(columns=['word','counts'])

for k,v in enumerate(l):
    if v not in bag:
        bag.append(pd.DataFrame(['ddd',2], columns = ['word','counts']))
print(bag)
# print('需要匹配的字符串：',b.group(1))