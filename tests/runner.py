# -*- coding: utf-8 -*-
# @Time    : 6/30/21 5:43 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import unittest


loader = unittest.TestLoader()
tests = loader.discover('.')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)

